from matplotlib.pyplot import show
from .base_trainer import BaseTrainer
from utils.misc import AverageMeter
from utils.misc import log
from utils.visualization_utils import plot_validation_fig, plot_training_fig, plot_image, plot_images, plot_warped_img, plot_imgs_and_lms, disp_warped_img, disp_training_fig
from utils.flow_utils import flow_warp, evaluate_flow, resize_flow_tensor
import numpy as np
from scipy.ndimage.interpolation import zoom as zoom
import torch
import time
from PIL import Image


class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func, args):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, args)

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()

        key_meter_names = ['Loss', 'l_ph', 'l_sm', "l_admm", "flow_mean", "flow_median"]
        key_meters = AverageMeter(i=len(key_meter_names),names=key_meter_names, precision=6)

        # self._validate()
        # puts the model in train mode
        self.model.train()
        end = time.time()

        for i_step, data in enumerate(self.train_loader):
            if i_step > self.args.epoch_size:
                break

            # Prepare data
            if isinstance(data,dict):
                img1, img2 = data['imgs']
            else: 
                img1, img2, case = data
            vox_dim =torch.cat([v[:,None] for v in img1[1]],dim=1).to(self.rank)
            img1, img2 = [im[0].to(self.rank) for im in [img1, img2]]
            if len(img1.shape) == 4:
                img1, img2 = [im.unsqueeze(1).float() for im in [img1, img2]]
            else:
                img1, img2 = [im.float() for im in [img1, img2]]
            
            
            # measure data loading time
            am_data_time.update(time.time() - end)

            # forward pass
            res_dict = self.model(img1, img2, vox_dim=vox_dim)
            flows12, flows21 = res_dict['flows_fw'][0], res_dict['flows_bk'][0]
            aux12, aux21 = res_dict['flows_fw'][1], res_dict['flows_bk'][1]
            
            flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                     zip(flows12, flows21)]
            aux = (aux12, aux21)

            loss, l_ph, l_sm, l_admm, flow_mean, occ_masks = self.loss_modules['loss_module'](flows, img1, img2, aux, vox_dim)

            # compute loss terms
            for loss_, module_ in self.loss_modules.items():
                
                # compute mwl loss 
                if "mwl" in loss_:
                    mask1, mask2 = [m[0].unsqueeze(1).to(self.rank) for m in data['target']['masks']]
                    l_mwl = module_(flows, mask1, mask2, occ_masks)
                    loss += l_mwl
                else: 
                    l_mwl = 0.0
                
                # compute cyclic loss
                if 'cyc' in loss_:
                    l_cyc = module_(flows, occ_masks, vox_dim)
                    loss += l_cyc
                else:
                    l_cyc = 0.0

                # compute kpts supervision
                if 'kpts' in loss_:
                    kpts = data['target']['kpts']
                    l_kpts = module_(flows[0],kpts, vox_dim)
                    loss += l_kpts
                else:
                    l_kpts = 0.0

            # update meters
            meters = [loss, l_ph, l_sm, l_admm, flow_mean, torch.median(torch.abs(flows12[0]))]
            vals = [m.item() if torch.is_tensor(m) else m for m in meters]
            key_meters.update(vals, img1.size(0))
            
            # compute gradient and do optimization step
            loss = loss.mean()
            self.loss = loss
            if np.isnan(loss.item()):
                print("here")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            am_batch_time.update(time.time() - end)
            end = time.time()

            if self.rank ==0 and self.i_iter % self.args.record_freq == 0:
                for v, name in zip(key_meters.val, key_meter_names):
                    self.summary_writer.add_scalar('Train_' + name, v, self.i_iter)
                    self.summary_writer.add_text('Hyperparams', self.dict2mdtable(self.args), 1)

            if self.rank == 0 and self.i_iter % self.args.print_freq == 0:
                istr = '{}:{}/{:4d}'.format(
                    self.i_epoch, i_step, self.epochs) + \
                       ' Time {} Data {}'.format(am_batch_time, am_data_time) + \
                       ' Info {}'.format(key_meters) + \
                       ' max flow {}'.format(torch.max(torch.abs(flows12[0]))) + \
                       ' min flow {}'.format(torch.min(torch.abs(flows12[0]))) 
                self._log.info(istr)

            
            #required_grad_params = [
            #    p for p in self.model.parameters() if p.requires_grad]
            #mean_grad_norm = 0
            #for param in [p for p in self.model.parameters() if p.requires_grad]:
            #    mean_grad_norm += param.grad.data.mean()

            self.i_iter += 1
            # break
        
        # to save parameters to tensor board 
    def dict2mdtable(self, d, key='Name', val='Value'):
        rows = [f'| {key} | {val} |']
        rows += ['|--|--|']
        rows += [f'| {k} | {v} |' for k, v in d.items()]
        return "  \n".join(rows)

    @ torch.no_grad()
    def _validate(self):
        self._log.info(f'Running validation on rank {self.rank}..')
        if hasattr(self.args,'dump_disp') and self.args.dump_disp:
            return self._dumpt_disp_fields()
        else:
            if self.args.valid_type == 'DTI_Example_valid':
                return self.DTI_validate()
            if self.args.valid_type == 'synthetic':
                return self.synt_validate()
            elif self.args.valid_type == 'variance_valid':
                return self.variance_validate()
            elif self.args.valid_type == 'l2r_valid':
                #return self._validate_with_gt() ##! I CHANGED IT HILA 
                return self.CT_validate()


    def _dumpt_disp_fields(self):
        dump_path = self.save_root  / "submission" / "task_02"
        if not self.args.docker:
            dump_path.mkdir(parents=True)
        self._log.info(f'rank {self.rank} - Dumping disp fields to {dump_path}')
        batch_time = AverageMeter()
        self.model.eval()
        end = time.time()

        all_error_names = []
        all_error_avgs = []

        error_names = ['TRE', 'LogJacDetStd']
        error_meters = AverageMeter(i=len(error_names))
        for i_step, data in enumerate(self.valid_loader):
            img1, img2 = data['imgs']
            if 'kpts' in data['target'].keys() and 'masks' in data['target'].keys():
                kpts, masks =  data['target']['kpts'], data['target']['masks']
            vox_dim =torch.cat([v[:,None] for v in img1[1]],dim=1).to(self.rank)
            img1, img2 = [im[0].to(self.rank) for im in [img1, img2]]
            if len(img1.shape) == 4:
                img1, img2 = [im.unsqueeze(1).float() for im in [img1, img2]]
            else:
                img1, img2 = [im.float() for im in [img1, img2]]
            
            # compute output
            flows = self.model(img2, img1, vox_dim=vox_dim, w_bk=False)['flows_fw'][0][0]
            pred_flows = flows.detach().squeeze(0)
            spacing = vox_dim.detach()

            # measure errors
            if 'kpts' in data['target'].keys() and 'masks' in data['target'].keys():
                es = evaluate_flow(pred_flows, kpts, masks, spacing)
            else:
                es = torch.tensor([-1,-1])
            error_meters.update([l.item() for l in es], img1.size(0))
            
            # dump flow to file
            disp_fields = resize_flow_tensor(pred_flows, shape=self.args.orig_shape)
            _, case_id = data['target']['case'][0].split('_')
            
            if not self.args.docker:
                filename = f'disp_{int(case_id):04}_{int(case_id):04}.npy'
                np.save(dump_path / filename, disp_fields.squeeze(0).cpu().numpy().astype(np.float32))
            else:
                filename = f'disp_{int(case_id):04}_{int(case_id):04}.npz'
                disp_fields = disp_fields.squeeze(0).cpu().numpy().astype(np.float32)
                disp_x = zoom(disp_fields[0], 0.5, order=2).astype('float16')
                disp_y = zoom(disp_fields[1], 0.5, order=2).astype('float16')
                disp_z = zoom(disp_fields[2], 0.5, order=2).astype('float16')
                disp = np.array((disp_x, disp_y, disp_z))

                # save displacement field
                np.savez_compressed(dump_path / filename, disp)
            
            # warped imgs
            img1_recons = flow_warp(img2, pred_flows.unsqueeze(0))
            p_warped = (disp_warped_img(img1[0].detach().cpu(), img1_recons[0].detach().cpu()).squeeze(0)*255).astype(np.uint8)
            p_valid, _ = disp_training_fig(img1[0].detach().cpu(), img2[0].detach().cpu(), pred_flows.cpu()).squeeze(0).transpose(1,2,0).astype(np.uint8)
            filename = f'warped_{int(case_id):04}_{int(case_id):04}.png'
            Image.fromarray(p_warped).save(dump_path / filename)
            filename = f'valid_{int(case_id):04}_{int(case_id):04}.png'
            Image.fromarray(p_valid).save(dump_path / filename)



            # measure elapsed time
            batch_time.update(time.time() - end)

            if i_step % self.args.print_freq == 0 or i_step == len(self.valid_loader) - 1:
                self._log.info('Test: [{0}/{1}]\t Time {2}\t '.format(i_step, len(self.valid_loader), batch_time) 
                    + ' '.join(map('{:.2f}'.format, error_meters.avg)))
            
            end = time.time()

        all_error_avgs.extend(error_meters.avg)
        all_error_names.extend(['{}'.format(name) for name in error_names])

        return all_error_avgs, all_error_names


        pass
    
    def _validate_with_gt(self):
        batch_time = AverageMeter()
        
        # only use the first GPU to run validation, multiple GPUs might raise error.
        # https://github.com/Eromera/erfnet_pytorch/issues/2#issuecomment-486142360
        #self.model = self.model.module
        self.model.eval()

        end = time.time()

        all_error_names = []
        all_error_avgs = []

        error_names = ['TRE', 'LogJacDetStd']
        error_meters = AverageMeter(i=len(error_names))
        for i_step, data in enumerate(self.valid_loader):
            img1, img2 = data['imgs']
            kpts, masks =  data['target']['kpts'], data['target']['masks']
            vox_dim =torch.cat([v[:,None] for v in img1[1]],dim=1).to(self.rank)
            img1, img2 = [im[0].to(self.rank) for im in [img1, img2]]
            img1, img2 = [im.unsqueeze(1).float() for im in [img1, img2]]
            
            # compute output
            flows = self.model(img1, img2, vox_dim=vox_dim, w_bk=False)['flows_fw'][0][0]
            pred_flows = flows.detach().squeeze(0)
            spacing = vox_dim.detach()

            # measure errors
            es = evaluate_flow(pred_flows, kpts, masks, spacing)
            error_meters.update([l.item() for l in es], img1.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)

            if i_step % self.args.print_freq == 0 or i_step == len(self.valid_loader) - 1:
                self._log.info('Test: [{0}/{1}]\t Time {2}\t '.format(i_step, len(self.valid_loader), batch_time) 
                    + ' '.join(map('{:.2f}'.format, error_meters.avg)))
            
            if i_step % self.args.plot_freq == 0:
                # 3d plots
                imgs = [img1, img2]
                figs = plot_imgs_and_lms(imgs, masks, kpts, pred_flows)
                self.summary_writer.add_figure('Valid_{}'.format(i_step), figs, self.i_epoch)
                # warped imgs
                img1_recons = flow_warp(img2[0].unsqueeze(0), pred_flows.unsqueeze(0))
                p_warped = disp_warped_img(img1[0].detach().cpu(), img1_recons[0].detach().cpu())
                #self.summary_writer.add_figure('Warped_{}'.format(i_step), p_warped, self.i_epoch)
                self.summary_writer.add_images('Warped_{}'.format(i_step), p_warped, self.i_epoch, dataformats='NHWC')
                # imgs and flow                
                p_valid, _ = disp_training_fig(img1[0].detach().cpu(), img2[0].detach().cpu(), pred_flows.cpu())
                self.summary_writer.add_images('Sample_{}'.format(i_step), p_valid, self.i_epoch, dataformats='NCHW')
                
                #p_valid = plot_images(img1[0].detach().cpu(), img1_recons[0].detach().cpu(),
                #                      img2[0].detach().cpu(), show=False)
                #self.writer.add_figure('Training_Images_warping_difference', p_valid, self.i_epoch)
                #diff_warp = torch.zeros([2, 192, 192, 64], device=self.device)
                #diff_warp[0] = img1[0]
                #diff_warp[1] = img1_recons[0]
                #diff_variance = torch.std(diff_warp, dim=0)
                #diff_error = float(diff_variance.median().item())
                #self.writer.add_scalar('Training error', diff_error,
                #                       self.i_iter)
                

            end = time.time()


        # write error to tf board.
        for value, name in zip(error_meters.avg, error_names):
            self.summary_writer.add_scalar(
                'Valid_{}'.format(name), value, self.i_epoch)

        all_error_avgs.extend(error_meters.avg)
        all_error_names.extend(['{}'.format(name) for name in error_names])

       # self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        # In order to reduce the space occupied during debugging,
        # only the model with more than cfg.save_iter iterations will be saved.
        if self.i_iter > self.args.save_iter:
            self.save_model(all_error_avgs[0], name=self.model_suffix)

        return all_error_avgs, all_error_names
    
    def synt_validate(self):
        error = 0
        loss = 0

        for i_step, data in enumerate(self.valid_loader):
            # torch.cuda.empty_cache()

            # Prepare data
            img1, img2, flow12 = data
            vox_dim = img1[1].to(self.device)
            img1, img2, flow12 = img1[0].to(self.device), img2[0].to(
                self.device), flow12[0].to(self.device)
            img1 = img1.unsqueeze(1).float().to(
                self.device)  # Add channel dimension
            img2 = img2.unsqueeze(1).float().to(
                self.device)  # Add channel dimension

            output = self.model(img1, img2, vox_dim=vox_dim)

            log(f'flow_size = {output[0].size()}')
            log(f'flow_size = {output[0].shape}')

            flow12_net = output[0].squeeze(0).float().to(
                self.device)  # Remove batch dimension, net prediction
            epe_map = torch.sqrt(
                torch.sum(torch.square(flow12 - flow12_net), dim=0)).to(self.device).mean()
            # epe_map = torch.abs(flow12 - flow12_net).to(self.device).mean()
            error += float(epe_map.mean().item())
            log(error)

            _loss, l_ph, l_sm = self.loss_func(output, img1, img2, vox_dim)
            loss += float(_loss.mean().item())
            # break

        error /= len(self.valid_loader)
        loss /= len(self.valid_loader)
        print(f'Validation error -> {error}')
        print(f'Validation loss -> {loss}')

        self.writer.add_scalar('Validation Error',
                               error,
                               self.i_epoch)

        self.writer.add_scalar('Validation Loss',
                               loss,
                               self.i_epoch)

        # p_imgs = [plot_image(im.detach().cpu(), show=False) for im in [img1, img2]]
        # p_conc_imgs= np.concatenate((np.concatenate(p_imgs[0][:1]+p_imgs[1][:1]),p_imgs[0][2]+p_imgs[1][2]))[np.newaxis][np.newaxis]
        # p_flows = [plot_flow(fl.detach().cpu(), show=False) for fl in [flow12,flow12_net]]
        # p_flows_conc = np.transpose(np.concatenate((np.concatenate(p_flows[0][:1]+p_flows[1][:1]),)),(2,0,1))[np.newaxis]
        # self.writer.add_images('Valid_Images_{}'.format(self.i_epoch), p_conc_imgs, self.i_epoch)
        # self.writer.add_images('Valid_Flows_{}'.format(self.i_epoch), p_flows_conc, self.i_epoch)

        # p_img_fig = plot_images(img1.detach().cpu(), img2.detach().cpu())
        # p_flo_gt = plot_flow(flow12.detach().cpu())
        # p_flo = plot_flow(flow12_net.detach().cpu())
        # self.writer.add_figure('Valid_Images_{}'.format(self.i_epoch), p_img_fig, self.i_epoch)
        # self.writer.add_figure('Valid_Flows_gt_{}'.format(self.i_epoch), p_flo_gt, self.i_epoch)
        # self.writer.add_figure('Valid_Flows_{}'.format(self.i_epoch), p_flo, self.i_epoch)

        p_valid = plot_validation_fig(img1.detach().cpu(), img2.detach().cpu(), flow12.detach().cpu(),
                                      flow12_net.detach().cpu(), show=False)
        self.writer.add_figure('Valid_Images', p_valid, self.i_epoch)

        return error, loss

    @torch.no_grad()
    def variance_validate(self):
        error_median = 0
        error_mean = 0
        error_short = 0
        max_diff_error = 0
        frame_diff_error = 0
        error_median_box = 0
        error_mean_box = 0
        error_short_box = 0
        max_diff_error_box = 0
        frame_diff_error_box = 0
        loss = 0
        im_h = im_w = 192
        im_d = 64
        flows = torch.zeros([3, im_h, im_w, im_d], device=self.device)
        images_warped = torch.zeros(
            [self.args.variance_valid_len, im_h, im_w, im_d], device=self.device)

        for i_step, data in enumerate(self.valid_loader):

            # Prepare data
            img1, img2, name = data
            vox_dim = img1[1].to(self.device)
            img1, img2 = img1[0].to(self.device), img2[0].to(self.device)
            img1 = img1.unsqueeze(1).float()  # Add channel dimension
            img2 = img2.unsqueeze(1).float()  # Add channel dimension

            if i_step % (self.args.variance_valid_len - 1) == 0:
                image0 = img1
                images_warped[i_step %
                              (self.args.variance_valid_len - 1)] = img1.squeeze(0)
                count = 0

            # Remove batch dimension, net prediction
            res = self.model(img1, img2, vox_dim=vox_dim, w_bk=False)[
                'flows_fw'][0][0].squeeze(0).float()
            flows += res
            # print(name)
            images_warped[(i_step % (self.args.variance_valid_len - 1))+1] = flow_warp(img2,
                                                                                   flows.unsqueeze(0))  # im1 recons
            count += 1

            if count == self.args.variance_valid_short_len - 1:
                variance = torch.std(images_warped[:count + 1, :, :, :], dim=0)
                error_short += float(variance.mean().item())
                box_variance = variance[49:148, 49:148, 16:48]
                error_short_box += float(box_variance.mean().item())

                log(error_short)
            if count == self.args.frame_dif+1:
                # calculating variance based only on model
                res = self.model(image0, img2, vox_dim=vox_dim, w_bk=False)[
                                 'flows_fw'][0][0].squeeze(0).float()
                diff_warp_straight = torch.zeros(
                    [2, im_h, im_w, im_d], device=self.device)
                diff_warp_straight[0] = images_warped[0]
                diff_warp_straight[1] = flow_warp(img2, res.unsqueeze(0))
                diff_variance_straight = torch.std(diff_warp_straight, dim=0)
                frame_diff_error += float(diff_variance_straight.median().item())
                box_variance = diff_variance_straight[49:148, 49:148, 16:48]
                frame_diff_error_box += float(box_variance.mean().item())
            # if (i_step + 1) % (self.args.variance_valid_len - 1) == 0:
            if count == self.args.variance_valid_len - 1:
                # calculating max_diff variance
                diff_warp = torch.zeros(
                    [2, im_h, im_w, im_d], device=self.device)
                diff_warp[0] = images_warped[0]
                diff_warp[1] = images_warped[-1]
                diff_variance = torch.std(diff_warp, dim=0)
                max_diff_error += float(diff_variance.mean().item())
                box_variance = diff_variance[49:148, 49:148, 16:48]
                max_diff_error_box += float(box_variance.mean().item())
                
                variance = torch.std(images_warped, dim=0)
                error_median += float(variance.median().item())
                error_mean += float(variance.mean().item())
                box_variance = variance[49:148, 49:148, 16:48]
                error_mean_box += float(box_variance.mean().item())
                error_median_box += float(box_variance.median().item())
                log(error_mean)
                flows = torch.zeros([3, im_h, im_w, im_d], device=self.device)
                count = 0
            # torch.cuda.empty_cache()

        max_diff_error /= self.args.variance_valid_sets
        frame_diff_error /= self.args.variance_valid_sets
        error_median /= self.args.variance_valid_sets
        error_mean /= self.args.variance_valid_sets
        error_short /= self.args.variance_valid_sets

        max_diff_error_box /= self.args.variance_valid_sets
        frame_diff_error_box /= self.args.variance_valid_sets
        error_median_box /= self.args.variance_valid_sets
        error_mean_box /= self.args.variance_valid_sets
        error_short_box /= self.args.variance_valid_sets
        # loss /= len(self.valid_loader)
        print(
            f'Validation maxDiff error-> {max_diff_error}, Validation error mean -> {error_mean}, Validation error median -> {error_median} Short Validation error -> {error_short}')
        # print(f'Validation loss -> {loss}')

        self.writer.add_scalar('Validation Difference_Error',
                               max_diff_error,
                               self.i_epoch)
        self.writer.add_scalar('Validation frame_difference_Error',
                               frame_diff_error,
                               self.i_epoch)
        self.writer.add_scalar('Validation Error(mean)',
                               error_mean,
                               self.i_epoch)
        self.writer.add_scalar('Validation Error(median)',
                               error_median,
                               self.i_epoch)
        self.writer.add_scalar('Validation Short Error',
                               error_short,
                               self.i_epoch)

        self.writer.add_scalar('Validation Difference_Error_box',
                               max_diff_error_box,
                               self.i_epoch)
        self.writer.add_scalar('Validation frame_difference_Error_box',
                               frame_diff_error_box,
                               self.i_epoch)
        self.writer.add_scalar('Validation Error(mean)_box',
                               error_mean_box,
                               self.i_epoch)
        self.writer.add_scalar('Validation Error(median)_box',
                               error_median_box,
                               self.i_epoch)
        self.writer.add_scalar('Validation Short Error_box',
                               error_short_box,
                               self.i_epoch)
        # self.writer.add_scalar('Validation Loss',
        #                        loss,
        #                        self.i_epoch)

        p_valid = plot_images(images_warped[0].detach().cpu(
        ), images_warped[-1].detach().cpu(), img2.detach().cpu(), show=False)
        # p_valid = plot_image(variance.detach().cpu(), show=False)
        #                               flow12_net.detach().cpu(), show=False)
        self.writer.add_figure('Valid_Images_original', p_valid, self.i_epoch)
        p_dif_valid = plot_images(images_warped[0].detach().cpu(
        ), diff_warp[-1].detach().cpu(), img2.detach().cpu(), show=False)
        p_dif_col = plot_warped_img(images_warped[0].detach().cpu(
        ), images_warped[-1].detach().cpu())
        self.writer.add_figure('Valid_Images_warped', p_dif_col, self.i_epoch)

        return [error_median], ["error_median"]



    def DTI_validate(self):
        batch_time = AverageMeter()
        
        # only use the first GPU to run validation, multiple GPUs might raise error.
        # https://github.com/Eromera/erfnet_pytorch/issues/2#issuecomment-486142360
        #self.model = self.model.module
        self.model.eval()

        end = time.time()

        all_error_names = []
        all_error_avgs = []


        for i_step, data in enumerate(self.valid_loader):
            img1, img2 = data['imgs']
            vox_dim =torch.cat([v[:,None] for v in img1[1]],dim=1).to(self.rank)
            img1, img2 = [im[0].to(self.rank) for im in [img1, img2]]
            # check if input in the correct shape [Batch, ch, D ,W, H]
            if len(img1.shape) == 4:
                img1, img2 = [im.unsqueeze(1).float() for im in [img1, img2]]
            else:
                img1, img2 = [im.float() for im in [img1, img2]]

           
            # compute output
            flows = self.model(img1, img2, vox_dim=vox_dim, w_bk=False)['flows_fw'][0][0]
            pred_flows = flows.detach().squeeze(0)
            spacing = vox_dim.detach()
            #! Please selected the channel (out of 6) of the DTI you want to visualize 
            selected_DTI_channel = 0
            if i_step % self.args.plot_freq == 0:
                
                # warped imgs
                img1_recons = flow_warp(img2[0].unsqueeze(0), pred_flows.unsqueeze(0))
                p_warped = disp_warped_img(img1[0][selected_DTI_channel].detach().cpu(),
                                             img1_recons[0][selected_DTI_channel].detach().cpu())
                #self.summary_writer.add_figure('Warped_{}'.format(i_step), p_warped, self.i_epoch)
                self.summary_writer.add_images('Warped_{}'.format(i_step), p_warped, self.i_epoch, dataformats='NHWC')
                # imgs and flow                
                p_valid, simple_flow_view = disp_training_fig(img1[0][selected_DTI_channel].detach().cpu(), img2[0][selected_DTI_channel].detach().cpu(), pred_flows.cpu())
                self.summary_writer.add_images('Sample_{}'.format(i_step), p_valid, self.i_epoch, dataformats='NCHW')
                self.summary_writer.add_figure('simple_flow_{}'.format(i_step), simple_flow_view, self.i_epoch)

                p_valid = plot_images(img1[0][selected_DTI_channel].detach().cpu(), img1_recons[0][selected_DTI_channel].detach().cpu(),
                                     img2[0][selected_DTI_channel].detach().cpu(), show=False)
                self.summary_writer.add_figure('Training_Images_warping_difference', p_valid, self.i_epoch)
                #diff_warp = torch.zeros([2, 192, 192, 64], device=self.device)
                #diff_warp[0] = img1[0]
                #diff_warp[1] = img1_recons[0]
                #diff_variance = torch.std(diff_warp, dim=0)
                #diff_error = float(diff_variance.median().item())
                #self.writer.add_scalar('Training error', diff_error,
                #                       self.i_iter)
                

            end = time.time()



       # self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        # In order to reduce the space occupied during debugging,
        # only the model with more than cfg.save_iter iterations will be saved.
        # if self.i_iter > self.args.save_iter:
        #     self.save_model(all_error_avgs[0], name=self.model_suffix)

        return all_error_avgs, all_error_names


     #! COPY OF DTI FOR CT 
    def CT_validate(self):
        batch_time = AverageMeter()
        
        # only use the first GPU to run validation, multiple GPUs might raise error.
        # https://github.com/Eromera/erfnet_pytorch/issues/2#issuecomment-486142360
        #self.model = self.model.module
        self.model.eval()

        end = time.time()

        all_error_names = []
        all_error_avgs = []

        #! train and not validate 
        for i_step, data in enumerate(self.train_loader):
            img1, img2 = data['imgs']
            vox_dim =torch.cat([v[:,None] for v in img1[1]],dim=1).to(self.rank)
            img1, img2 = [im[0].to(self.rank) for im in [img1, img2]]
            # check if input in the correct shape [Batch, ch, D ,W, H]
            if len(img1.shape) == 4:
                img1, img2 = [im.unsqueeze(1).float() for im in [img1, img2]]
            else:
                img1, img2 = [im.float() for im in [img1, img2]]

           
            # compute output
            flows = self.model(img1, img2, vox_dim=vox_dim, w_bk=False)['flows_fw'][0][0]
            pred_flows = flows.detach().squeeze(0)
            spacing = vox_dim.detach()
           

            if i_step % self.args.plot_freq == 0:
                
                # warped imgs
                img1_recons = flow_warp(img2[0].unsqueeze(0), pred_flows.unsqueeze(0))
                p_warped = disp_warped_img(img1[0].detach().cpu(),
                                             img1_recons[0].detach().cpu())
                #self.summary_writer.add_figure('Warped_{}'.format(i_step), p_warped, self.i_epoch)
                self.summary_writer.add_images('Warped_{}'.format(i_step), p_warped, self.i_epoch, dataformats='NHWC')
                # imgs and flow                
                p_valid, simple_flow_view = disp_training_fig(img1[0].detach().cpu(), img2[0].detach().cpu(), pred_flows.cpu())
                self.summary_writer.add_images('Sample_{}'.format(i_step), p_valid, self.i_epoch, dataformats='NCHW')
                self.summary_writer.add_figure('simple_flow_{}'.format(i_step), simple_flow_view, self.i_epoch)

                p_valid = plot_images(img1[0].detach().cpu(), img1_recons[0].detach().cpu(),
                                     img2[0].detach().cpu(), show=False)
                self.summary_writer.add_figure('Training_Images_warping_difference', p_valid, self.i_epoch)

                

            end = time.time()



        return all_error_avgs, all_error_names
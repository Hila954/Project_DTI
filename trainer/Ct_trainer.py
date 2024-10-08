from matplotlib.pyplot import show
from .base_trainer import BaseTrainer
from utils.misc import AverageMeter
from utils.misc import log
from utils.visualization_utils import plot_validation_fig, plot_training_fig, plot_image, plot_images, plot_warped_img, plot_imgs_and_lms, disp_warped_img, disp_training_fig
from utils.flow_utils import flow_warp, evaluate_flow, resize_flow_tensor
from utils.distance_between_images import compute_distances_array, compute_dijkstra_validation
import numpy as np
from scipy.ndimage.interpolation import zoom as zoom
import torch
import time
from PIL import Image
from losses.NCC import NCC
from utils.distance_between_images import pick_points_in_DTI



class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func, args):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, args)


    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()

        key_meter_names = ['Loss', 'l_ph', 'l_sm', "l_cyc", "flow_mean", "flow_median", "remaining_voxels_percent"]
        key_meters = AverageMeter(i=len(key_meter_names),names=key_meter_names, precision=6)

        # self._validate()
        # puts the model in train mode
        self.model.train()
        end = time.time()

        for i_step, data in enumerate(self.train_loader):
            if i_step > self.args.epoch_size: ### pay attention 
                break
            
            if self.i_epoch == 1: # print only once 
                self._log.info('=> Training Data: {} vs {}'.format(self.args.first_animal_name, self.args.second_animal_name))

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
            remaining_voxels = (occ_masks[0][0] != 0).sum()
            occ_voxels = (occ_masks[0][0] == 0).sum()
            remaining_voxels_percent = remaining_voxels / (remaining_voxels + occ_voxels)
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
            meters = [loss, l_ph, l_sm, l_cyc, flow_mean, torch.median(torch.abs(flows12[0])), remaining_voxels_percent]
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

            if self.rank ==0 and self.i_epoch % self.args.record_freq == 0:
                for v, name in zip(key_meters.val, key_meter_names):
                    self.summary_writer.add_scalar('Train_' + name, v, self.i_epoch)
            
            # Occ Mask visual view 
            if self.i_epoch % self.args.binary_mask_freq == 0:
                binary_occ_mask1 = occ_masks[0][0].detach().cpu().squeeze()
                binary_occ_mask2 = occ_masks[0][1].detach().cpu().squeeze()
                img_binary_occ_mask1 = plot_image(binary_occ_mask1)
                img_binary_occ_mask2 = plot_image(binary_occ_mask2)
                self.summary_writer.add_figure(f'binary_mask_occ1', img_binary_occ_mask1, self.i_epoch)
                self.summary_writer.add_figure(f'binary_mask_occ2', img_binary_occ_mask2, self.i_epoch)


            if self.rank == 0 and self.i_epoch % self.args.print_freq == 0:
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
        
        # only use the first GPU to run validation, multiple GPUs might raise error.
        # https://github.com/Eromera/erfnet_pytorch/issues/2#issuecomment-486142360
        #self.model = self.model.module
        self.model.eval()

        all_error_names = []
        all_error_avgs = []


        for i_step, data in enumerate(self.valid_loader):
            self._log.info('=> Data: {} vs {}'.format(self.args.first_animal_name, self.args.second_animal_name))

            img1, img2 = data['imgs']
            vox_dim =torch.cat([v[:,None] for v in img1[1]],dim=1).to(self.rank)
            img1, img2 = [im[0].to(self.rank) for im in [img1, img2]]
            # check if input in the correct shape [Batch, ch, D ,W, H]
            if len(img1.shape) == 4:
                img1, img2 = [im.unsqueeze(1).float() for im in [img1, img2]]
            else:
                img1, img2 = [im.float() for im in [img1, img2]]

           
            # compute output
            flows = self.model(img1, img2, vox_dim=vox_dim, w_bk=True)
            flows_fw = flows['flows_fw'][0][0]
            flows_bk = flows['flows_bk'][0][0]

            pred_flows = flows_fw.detach().squeeze(0)
            pred_flows_bk = flows_bk.detach().squeeze(0)

            spacing = vox_dim.detach()
            if 'SHIFTED' in self.args.model_suffix: 
                GT_shift_value = self.valid_loader.dataset.GT_shift_value
                self._log.info(f'GT_shift_value={GT_shift_value}')
                GT_for_pixel_shift = torch.zeros_like(pred_flows)
                GT_for_pixel_shift_bk = torch.zeros_like(pred_flows)

                GT_for_pixel_shift[2, :, :, :] += GT_shift_value
                GT_for_pixel_shift_bk[2, :, :, :] += -1*GT_shift_value

                #GT_for_20pixel_shift[2,  self.valid_loader.dataset.none_zero_indexes] = -20
                
            
                #! MSE
                MSE = torch.mean((pred_flows[:, :, :, GT_shift_value:] - GT_for_pixel_shift[:, :, :, GT_shift_value:]) ** 2) 
                self.summary_writer.add_scalar('Validation_MSE', MSE, self.i_epoch)
                #! I think that for the back wrapping there are different occluded pixels so we should discard a different area, the code here is not accurate  
                MSE_bk = torch.mean((pred_flows_bk[:, :, :, GT_shift_value:] - GT_for_pixel_shift_bk[:, :, :, GT_shift_value:]) ** 2) 
                self.summary_writer.add_scalar('Validation_MSE_bk', MSE_bk, self.i_epoch)

            # warped imgs
            img1_recons = flow_warp(img2, pred_flows.unsqueeze(0))[0]
            img2_recons = flow_warp(img1, pred_flows_bk.unsqueeze(0))[0]

            img1 = img1[0]
            img2 = img2[0]
            


            #!  visualize all channels
            for selected_DTI_channel in range(img1.shape[0]):

                if i_step % self.args.plot_freq == 0:
                               
                    # VALIDATION METRICS
                    self._log.info(f'Running validation for ch {selected_DTI_channel}..')

                    # MSE 
                    MSE = torch.mean((img1[selected_DTI_channel] - img1_recons[selected_DTI_channel]) ** 2) 
                    self.summary_writer.add_scalar('Validation_MSE_ch_{}'.format(selected_DTI_channel), MSE, self.i_epoch)

                    MSE_bk = torch.mean((img2[selected_DTI_channel] - img2_recons[selected_DTI_channel]) ** 2)
                    self.summary_writer.add_scalar('Validation_MSE_bk_ch_{}'.format(selected_DTI_channel), MSE_bk, self.i_epoch)

                    #RMSE
                    RMSE = torch.sqrt(torch.mean((img1[selected_DTI_channel] - img1_recons[selected_DTI_channel]) ** 2)) 
                    self.summary_writer.add_scalar('Validation_RMSE_ch_{}'.format(selected_DTI_channel), RMSE, self.i_epoch)

                    RMSE_bk = torch.sqrt(torch.mean((img2[selected_DTI_channel] - img2_recons[selected_DTI_channel]) ** 2))
                    self.summary_writer.add_scalar('Validation_RMSE_bk_ch_{}'.format(selected_DTI_channel), RMSE_bk, self.i_epoch)

                    #MAE mean absolute error 
                    MAE = torch.mean(torch.abs(img1[selected_DTI_channel] - img1_recons[selected_DTI_channel])) 
                    self.summary_writer.add_scalar('Validation_MAE_ch_{}'.format(selected_DTI_channel), MAE, self.i_epoch)

                    MAE_bk = torch.mean(torch.abs(img2[selected_DTI_channel] - img2_recons[selected_DTI_channel]))
                    self.summary_writer.add_scalar('Validation_MAE_bk_ch_{}'.format(selected_DTI_channel), MAE_bk, self.i_epoch)
                    ###################################


                    p_warped = disp_warped_img(img1[selected_DTI_channel].detach().cpu(),
                                                img1_recons[selected_DTI_channel].detach().cpu())
                    #self.summary_writer.add_figure('Warped_{}'.format(i_step), p_warped, self.i_epoch)
                    self.summary_writer.add_images('Warped_ch_{}'.format(selected_DTI_channel), p_warped, self.i_epoch, dataformats='NHWC')
                    p_warped_bk = disp_warped_img(img2[selected_DTI_channel].detach().cpu(),
                                                img2_recons[selected_DTI_channel].detach().cpu())
                    self.summary_writer.add_images('Warped_ch_{}_bk'.format(selected_DTI_channel), p_warped_bk, self.i_epoch, dataformats='NHWC')

                    # imgs and flow                
                    p_valid, simple_flow_view = disp_training_fig(img1[selected_DTI_channel].detach().cpu(), img2[selected_DTI_channel].detach().cpu(), pred_flows.cpu())
                    self.summary_writer.add_images('Sample_ch_{}'.format(selected_DTI_channel), p_valid, self.i_epoch, dataformats='NCHW')

                    p_valid = plot_images(img1[selected_DTI_channel].detach().cpu(), img1_recons[selected_DTI_channel].detach().cpu(),
                                        img2[selected_DTI_channel].detach().cpu(), img2_recons[selected_DTI_channel].detach().cpu(), show=False)
                    self.summary_writer.add_figure(f'Training_Images_warping_difference_{selected_DTI_channel}', p_valid, self.i_epoch)
                    #diff_warp = torch.zeros([2, 192, 192, 64], device=self.device)
                    #diff_warp[0] = img1[0]
                    #diff_warp[1] = img1_recons[0]
                    #diff_variance = torch.std(diff_warp, dim=0)
                    #diff_error = float(diff_variance.median().item())
                    #self.writer.add_scalar('Training error', diff_error,
                    #                       self.i_iter)
                    
            self.summary_writer.add_figure('simple_flow_middle_slice', simple_flow_view, self.i_epoch)




       # self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        # In order to reduce the space occupied during debugging,
        # only the model with more than cfg.save_iter iterations will be saved.
        # if self.i_iter > self.args.save_iter:
        #     self.save_model(all_error_avgs[0], name=self.model_suffix)

        return all_error_avgs, all_error_names
    
    def _calculate_distance_between_DTI(self):
        self.model.eval()
        
        loss_ncc_func = NCC(win=self.args.ncc_win[0]) # we will take the first one as all levels have the same ncc win

        for i_step, data in enumerate(self.valid_loader):
            self._log.info('=> Data: {} vs {}'.format(self.args.first_animal_name, self.args.second_animal_name))

            img1, img2 = data['imgs']
            vox_dim1, vox_dim2 = img1[1].squeeze().numpy(), img2[1].squeeze().numpy()
            vox_dim =torch.cat([v[:,None] for v in img1[1]],dim=1).to(self.rank) #just take the first one in general, it is just scale in the gradient 
            
            self._log.info('=> vomdim1: {}, voxdim2 {}'.format(vox_dim1, vox_dim2))

            img1, img2 = [im[0].to(self.rank) for im in [img1, img2]]
            # check if input in the correct shape [Batch, ch, D ,W, H]
            if len(img1.shape) == 4:
                img1, img2 = [im.unsqueeze(1).float() for im in [img1, img2]]
            else:
                img1, img2 = [im.float() for im in [img1, img2]]


            # compute output
            flows = self.model(img1, img2, vox_dim=vox_dim, w_bk=True)
            flows_fw = flows['flows_fw'][0][0]
            flows_bk = flows['flows_bk'][0][0]

            pred_flows = flows_fw.detach().squeeze(0)
            pred_flows_bk = flows_bk.detach().squeeze(0)

            # warped imgs
            img1_recons = flow_warp(img2, pred_flows.unsqueeze(0))
            img2_recons = flow_warp(img1, pred_flows_bk.unsqueeze(0))

            # loss_ncc 
            loss_ncc_flow12 = loss_ncc_func(img1, img1_recons)
            loss_ncc_flow21 = loss_ncc_func(img2, img2_recons)

            img1, img2 = img1.cpu().numpy().squeeze(), img2.cpu().numpy().squeeze()
            img1_recons, img2_recons = img1_recons.cpu().numpy().squeeze(), img2_recons.cpu().numpy().squeeze()

            

            self._log.info(f'flow12 loss {loss_ncc_flow12}')
            self._log.info(f'flow21 loss {loss_ncc_flow21}')

            # choose points for the dijkstra algorithm, by finding points with values != None 
            how_many_points = self.args.how_many_points_for_dist
            chosen_lambda = self.args.lambda_distance
            win_len_for_distance = self.args.win_len_for_distance

            self._log.info(f'using {how_many_points} points ')
            self._log.info(f'lambda is {chosen_lambda}')
            self._log.info(f'win_len_for_distance is {win_len_for_distance}')

            if loss_ncc_flow21 >= loss_ncc_flow12:
                self._log.info('flow12 smaller or equal')
                points_array1 = pick_points_in_DTI(img1, how_many_points) # in relation to img1 

                brain_distance_flow12 = compute_distances_array(img1, img2, pred_flows, points_array1, vox_dim1, vox_dim2,
                                                                 chosen_lambda, win_len_for_distance)
                direction = f'{self.args.first_animal_name} to {self.args.second_animal_name} flow12'

                self._log.info(f'brain_distance {direction} is {brain_distance_flow12}')
                final_distance = brain_distance_flow12
            else:
                self._log.info('flow21 smaller')

                points_array2 = pick_points_in_DTI(img2, how_many_points) # in relation to img2

                
                brain_distance_flow21 = compute_distances_array(img2, img1, pred_flows_bk, points_array2, vox_dim2, vox_dim1,
                                                                 chosen_lambda, win_len_for_distance)
                direction = f'{self.args.second_animal_name} to {self.args.first_animal_name} flow21'

                self._log.info(f'brain_distance {direction} is {brain_distance_flow21}')
                final_distance = brain_distance_flow21

            
            return final_distance






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
import imp
import numpy as np
import os
import datetime
import time
import torch
import torch.nn as nn
from ops.Logger import AverageMeter, ProgressMeter
from data_processing.DRNA_dataset import Single_Dataset2
from model.Small_Unet_3Plus_DeepSup import Small_UNet_3Plus_DeepSup
from tqdm.auto import tqdm
import sys


def gen_input_data(map_data, chain_prob, base_prob, voxel_size, stride, contour, train_save_path):
    from progress.bar import Bar
    scan_x, scan_y, scan_z = map_data.shape
    chain_classes = len(chain_prob)
    base_classes = len(base_prob)
    count_voxel = 0
    count_iter = 0
    Coord_Voxel = []
    bar = Bar('Preparing Input: ',
              max=int(np.ceil(scan_x / stride) * np.ceil(scan_y / stride) * np.ceil(scan_z / stride)))

    for x in range(0, scan_x, stride):
        x_end = min(x + voxel_size, scan_x)
        for y in range(0, scan_y, stride):
            y_end = min(y + voxel_size, scan_y)
            for z in range(0, scan_z, stride):
                bar.next()
                count_iter += 1
                z_end = min(z + voxel_size, scan_z)
                if x_end < scan_x:
                    x_start = x
                else:
                    x_start = x_end - voxel_size
                    if x_start < 0:
                        x_start = 0
                if y_end < scan_y:
                    y_start = y
                else:
                    y_start = y_end - voxel_size
                    if y_start < 0:
                        y_start = 0
                if z_end < scan_z:
                    z_start = z
                else:
                    z_start = z_end - voxel_size
                    if z_start < 0:
                        z_start = 0
                # already normalized
                segment_map_voxel = map_data[x_start:x_end, y_start:y_end, z_start:z_end]
                if contour <= 0:
                    meaningful_density_count = len(np.argwhere(segment_map_voxel > 0))
                    meaningful_density_ratio = meaningful_density_count / float(voxel_size ** 3)
                    if meaningful_density_ratio <= 0.001:
                        # print("meaningful density ratio %f of current box, skip it!"%meaningful_density_ratio)
                        continue
                else:
                    meaningful_density_count = len(np.argwhere(segment_map_voxel > contour))
                    meaningful_density_ratio = meaningful_density_count / float(voxel_size ** 3)
                    if meaningful_density_ratio <= 0.001:
                        # print("no meaningful density ratio of current box, skip it!")
                        continue
                segment_input_voxel = np.zeros([chain_classes + base_classes, voxel_size, voxel_size, voxel_size])
                segment_input_voxel[:chain_classes, :x_end - x_start, :y_end - y_start, :z_end - z_start] = chain_prob[
                                                                                                            :,
                                                                                                            x_start:x_end,
                                                                                                            y_start:y_end,
                                                                                                            z_start:z_end]
                segment_input_voxel[chain_classes:, :x_end - x_start, :y_end - y_start, :z_end - z_start] = base_prob[:,
                                                                                                            x_start:x_end,
                                                                                                            y_start:y_end,
                                                                                                            z_start:z_end]
                # check values in segment input_voxel
                # different classes >=0.5 number should be bigger than 0.001
                count_meaningful = (segment_input_voxel > 0.5).sum()
                meaningful_density_ratio = count_meaningful / float(voxel_size ** 3)
                if meaningful_density_ratio <= 0.001:
                    # print("no meaningful predictions of current box in 1st stage, skip it!")
                    continue

                cur_path = os.path.join(train_save_path, "input_" + str(count_voxel) + ".npy")
                np.save(cur_path, segment_input_voxel)
                Coord_Voxel.append([x_start, y_start, z_start])
                count_voxel += 1
                # print("2nd stage: %.2f percent scanning finished"%(count_iter/(scan_x*scan_y*scan_z/(stride**3))))
    bar.finish()
    Coord_Voxel = np.array(Coord_Voxel)
    coord_path = os.path.join(train_save_path, "Coord.npy")
    np.save(coord_path, Coord_Voxel)
    print("in 2nd stage, in total we have %d boxes" % len(Coord_Voxel))
    return Coord_Voxel


from model.Small_Unet_3Plus_DeepSup import Small_UNet_3Plus_DeepSup
import gc


def make_predictions(test_loader, model, Coord_Voxel, voxel_size, overall_shape, num_classes, run_type=0):
    model.eval()
    device = next(model.parameters()).device
    scan_x, scan_y, scan_z = overall_shape

    # Initialize prediction matrix on GPU
    prediction_matrix = torch.zeros((num_classes, *overall_shape), device=device)

    # Initialize timing meters
    avg_meters = {
        'data_time': AverageMeter('data_time'),
        'train_time': AverageMeter('train_time')
    }

    def get_bounds(coord, size, max_size):
        end = min(coord + size, max_size)
        start = max(end - size, 0)
        return start, end

    def get_activation(outputs):
        if run_type == 2:
            return torch.softmax(torch.sigmoid(outputs[0]), dim=1)
        elif run_type == 1:
            return torch.sigmoid(outputs[0])
        return torch.softmax(outputs[0], dim=1)

    # Progress bar setup
    total = len(test_loader)
    pbar = tqdm(total=total, dynamic_ncols=True, position=0, leave=True,
                desc="Predicting", file=sys.stdout)
    end_time = time.time()

    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_idx, (inputs, cur_index) in enumerate(test_loader):
            # Timing update
            avg_meters['data_time'].update(time.time() - end_time)

            # Move inputs to GPU
            inputs = inputs.to(device)
            cur_id = cur_index.cpu().numpy()

            # Get model predictions
            outputs = model(inputs)
            final_output = get_activation(outputs)

            # Process each sample in batch
            for k, idx in enumerate(cur_id):
                coords = Coord_Voxel[int(idx)]

                # Get coordinate bounds
                x_start, x_end = get_bounds(int(coords[0]), voxel_size, scan_x)
                y_start, y_end = get_bounds(int(coords[1]), voxel_size, scan_y)
                z_start, z_end = get_bounds(int(coords[2]), voxel_size, scan_z)

                # Update predictions using torch.maximum
                prediction_matrix[:, x_start:x_end, y_start:y_end, z_start:z_end] = torch.maximum(
                    prediction_matrix[:, x_start:x_end, y_start:y_end, z_start:z_end],
                    final_output[k, :, :x_end - x_start, :y_end - y_start, :z_end - z_start]
                )

            # Update timing
            avg_meters['train_time'].update(time.time() - end_time)

            # Update progress bar with timing info
            pbar.set_postfix({
                'data_time': f"{avg_meters['data_time'].avg:.3f}",
                'train_time': f"{avg_meters['train_time'].avg:.3f}"
            })
            pbar.update(1)

            # Print detection stats periodically
            if batch_idx % 1000 == 0:
                for j in range(num_classes):
                    count_positive = (prediction_matrix[j] >= 0.5).sum().item()
                    print(f"\nClass {j} detected {count_positive} voxels", flush=True)

            end_time = time.time()

            # Clean up batch data
            del outputs, final_output

    pbar.close()

    # Final processing - move to CPU at the end
    prediction_matrix = prediction_matrix.cpu().numpy()
    prediction_matrix = np.nan_to_num(prediction_matrix, 0)
    prediction_label = np.argmax(prediction_matrix, axis=0)

    return prediction_matrix, prediction_label


def unet_detect_map_refine(map_data, chain_prob, base_prob, resume_model_path, voxel_size,
                           stride, batch_size, train_save_path, contour, params):
    coord_path = os.path.join(train_save_path, "Coord.npy")
    if os.path.exists(coord_path):
        Coord_Voxel = np.load(coord_path)
    else:
        Coord_Voxel = gen_input_data(map_data, chain_prob, base_prob, voxel_size, stride, contour, train_save_path)
    overall_shape = map_data.shape
    test_dataset = Single_Dataset2(train_save_path, "input_")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=params['num_workers'],
        drop_last=False)
    chain_class = len(chain_prob)
    base_class = len(base_prob)
    output_classes = chain_class + base_class
    model = Small_UNet_3Plus_DeepSup(in_channels=chain_class + base_class,
                                     n_classes=output_classes,
                                     feature_scale=4,
                                     is_deconv=True,
                                     is_batchnorm=True)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    state_dict = torch.load(resume_model_path)
    msg = model.load_state_dict(state_dict['state_dict'])
    print("model loading: ", msg)
    cur_prob_path = os.path.join(train_save_path, "chain_predictprob.npy")
    cur_label_path = os.path.join(train_save_path, "chain_predict.npy")
    if os.path.exists(cur_prob_path) and os.path.exists(cur_label_path):
        Prediction_Matrix = np.load(cur_prob_path)
        Prediction_Label = np.load(cur_label_path)

    else:
        Prediction_Matrix, Prediction_Label = make_predictions(test_loader, model, Coord_Voxel,
                                                               voxel_size, overall_shape,
                                                               output_classes, run_type=1)  # must sigmoid activated

        np.save(cur_prob_path, Prediction_Matrix)
        np.save(cur_label_path, Prediction_Label)
    return Prediction_Matrix

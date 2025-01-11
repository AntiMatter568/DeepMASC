import numpy as np
import os
import datetime
import time
import torch
import torch.nn as nn
from ops.Logger import AverageMeter, ProgressMeter
from data_processing.DRNA_dataset import Single_Dataset
from model.Cascade_Unet import Cascade_Unet
from tqdm.auto import tqdm
import sys


def gen_input_data(map_data, voxel_size, stride, contour, train_save_path):
    scan_x, scan_y, scan_z = map_data.shape
    count_voxel = 0
    count_iter = 0
    Coord_Voxel = []
    from progress.bar import Bar
    bar = Bar('Preparing Input: ',
              max=int(np.ceil(scan_x / stride) * np.ceil(scan_y / stride) * np.ceil(scan_z / stride)))

    for x in range(0, scan_x, stride):
        x_end = min(x + voxel_size, scan_x)
        for y in range(0, scan_y, stride):
            y_end = min(y + voxel_size, scan_y)
            for z in range(0, scan_z, stride):
                count_iter += 1
                bar.next()
                # print("1st stage: %.4f percent scanning finished"%(count_iter*100/(scan_x*scan_y*scan_z/(stride**3))),"location %d %d %d"%(x,y,z))
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
                segment_map_voxel = np.zeros([voxel_size, voxel_size, voxel_size])
                segment_map_voxel[:x_end - x_start, :y_end - y_start, :z_end - z_start] = map_data[x_start:x_end,
                                                                                          y_start:y_end, z_start:z_end]
                if contour <= 0:
                    meaningful_density_count = len(np.argwhere(segment_map_voxel > 0))
                    meaningful_density_ratio = meaningful_density_count / float(voxel_size ** 3)
                    if meaningful_density_ratio <= 0.001:
                        # print("no meaningful density ratio %f in current scanned box, skip it!"%meaningful_density_ratio)
                        continue
                else:
                    meaningful_density_count = len(np.argwhere(segment_map_voxel > contour))
                    meaningful_density_ratio = meaningful_density_count / float(voxel_size ** 3)
                    if meaningful_density_ratio <= 0.001:
                        # print("no meaningful density ratio in current scanned box, skip it!")
                        continue
                cur_path = os.path.join(train_save_path, "input_" + str(count_voxel) + ".npy")
                np.save(cur_path, segment_map_voxel)
                Coord_Voxel.append([x_start, y_start, z_start])
                count_voxel += 1
    bar.finish()
    Coord_Voxel = np.array(Coord_Voxel)
    coord_path = os.path.join(train_save_path, "Coord.npy")
    np.save(coord_path, Coord_Voxel)
    print("In total we prepared %d boxes as input" % (len(Coord_Voxel)))
    return Coord_Voxel


def make_predictions(test_loader, model, Coord_Voxel, voxel_size, overall_shape, num_classes, base_classes):
    model.eval()
    scan_x, scan_y, scan_z = overall_shape

    # Initialize matrices on GPU to avoid CPU-GPU transfers
    device = next(model.parameters()).device
    prediction_matrix = torch.zeros((num_classes, *overall_shape), device=device)
    base_matrix = torch.zeros((base_classes, *overall_shape), device=device)
    count_matrix = torch.zeros(overall_shape, device=device)

    def get_bounds(coord, size, max_size):
        end = min(coord + size, max_size)
        start = max(end - size, 0)
        return start, end

    # Improved progress bar setup
    total = len(test_loader)
    pbar = tqdm(total=total, dynamic_ncols=True, position=0, leave=True,
                desc="Predicting", file=sys.stdout)

    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_idx, (input_data, cur_index) in enumerate(test_loader):
            input_data = input_data.to(device)
            cur_id = cur_index.cpu().numpy()

            # Get model predictions
            chain_outputs, base_outputs = model(input_data)
            final_output = torch.sigmoid(chain_outputs[0])
            final_base = torch.sigmoid(base_outputs[0])

            # Update matrices batch-wise
            for k, idx in enumerate(cur_id):
                coords = Coord_Voxel[int(idx)]

                # Get coordinate bounds
                x_start, x_end = get_bounds(int(coords[0]), voxel_size, scan_x)
                y_start, y_end = get_bounds(int(coords[1]), voxel_size, scan_y)
                z_start, z_end = get_bounds(int(coords[2]), voxel_size, scan_z)

                # Slice views for efficiency
                x_slice = slice(x_start, x_end)
                y_slice = slice(y_start, y_end)
                z_slice = slice(z_start, z_end)

                # Update predictions directly on GPU
                prediction_matrix[:, x_slice, y_slice, z_slice] += final_output[k, :,
                                                                   :x_end - x_start, :y_end - y_start, :z_end - z_start]
                base_matrix[:, x_slice, y_slice, z_slice] += final_base[k, :,
                                                             :x_end - x_start, :y_end - y_start, :z_end - z_start]
                count_matrix[x_slice, y_slice, z_slice] += 1

            # Update progress bar
            pbar.update(1)

            # Log detections periodically
            if batch_idx % 1000 == 0:
                with torch.no_grad():
                    detections = [(j, (prediction_matrix[j] >= 0.5).sum().item())
                                  for j in range(num_classes)]
                    pbar.set_postfix({'Detected': detections}, refresh=True)
                    print("\nDetection counts:", flush=True)
                    for j, count in detections:
                        print(f"Class {j}: {count} voxels", flush=True)

    pbar.close()

    # Final computations on GPU
    count_matrix = count_matrix.unsqueeze(0)  # Add channel dimension for broadcasting
    prediction_matrix = prediction_matrix / count_matrix
    base_matrix = base_matrix / count_matrix

    # Handle NaN values
    prediction_matrix = torch.nan_to_num(prediction_matrix, 0)
    base_matrix = torch.nan_to_num(base_matrix, 0)

    # Move to CPU only at the end
    prediction_matrix = prediction_matrix.cpu().numpy()
    base_matrix = base_matrix.cpu().numpy()

    # Final processing
    prediction_label = np.argmax(prediction_matrix, axis=0)
    new_base_matrix = base_matrix[:-1].copy()
    new_base_matrix[1] = np.maximum(base_matrix[1], base_matrix[-1])
    base_label = np.argmax(new_base_matrix, axis=0)

    return prediction_matrix, prediction_label, new_base_matrix, base_label


def unet_detect_map_cascad(map_data, resume_model_path, voxel_size,
                           stride, batch_size, train_save_path, contour, params):
    coord_path = os.path.join(train_save_path, "Coord.npy")
    if os.path.exists(coord_path):
        Coord_Voxel = np.load(coord_path)
    else:
        Coord_Voxel = gen_input_data(map_data, voxel_size, stride, contour, train_save_path)
    overall_shape = map_data.shape
    test_dataset = Single_Dataset(train_save_path, "input_")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=params['num_workers'],
        drop_last=False)
    chain_class = 4
    base_class = 5
    model = Cascade_Unet(in_channels=1,  # include density and probability array
                         n_classes1=chain_class,
                         n_classes2=base_class,
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
    cur_baseprob_path = os.path.join(train_save_path, "base_predictprob.npy")
    cur_baselabel_path = os.path.join(train_save_path, "base_predict.npy")
    if os.path.exists(cur_prob_path) and os.path.exists(cur_label_path):
        Prediction_Matrix = np.load(cur_prob_path)
        Prediction_Label = np.load(cur_label_path)
        Base_Matrix = np.load(cur_baseprob_path)
        Base_Label = np.load(cur_baselabel_path)
    else:
        Prediction_Matrix, Prediction_Label, Base_Matrix, Base_Label = make_predictions(test_loader, model, Coord_Voxel,
                                                                                        voxel_size, overall_shape,
                                                                                        chain_class, base_class)

        np.save(cur_prob_path, Prediction_Matrix)
        np.save(cur_label_path, Prediction_Label)
        np.save(cur_baseprob_path, Base_Matrix)
        np.save(cur_baselabel_path, Base_Label)
    # save disk space for the generated input boxes
    os.system("rm " + train_save_path + "/input*")
    return Prediction_Matrix, Base_Matrix

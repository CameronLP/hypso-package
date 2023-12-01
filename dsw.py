"Translated from Matlab to C back to Python by SBO 22.11.2023"
import numpy as np

import math

def novel_dsw(con, R, C):
   
    height = con['height']
    width = con['width']
    dif = con['win_dif']
    size = con['win_size']
    code = con['code_size']

    win_total = size + 2 * dif # height and width if inner window.
    neighbours = (win_total * win_total - size * size)
    under_halfway = dif + (size - 1) // 2
    BOUNDARY_BOTTOM_RIGHT = dif
    BOUNDARY_TOP_LEFT = win_total - dif
    WIN_LENGTH = win_total
    NEIGHBOURS = ( (win_total)*(win_total) - (size*size) )

    #define UNDER_HALFWAY (WIN_DIF + (WIN_SIZE - 1)/2)
    #define OVER_HALFWAY (WIN_SIZE + (WIN_DIF * 2) - UNDER_HALFWAY)
    #define WIN_TOTAL (WIN_LENGTH)
    #define VALID_TOP HEIGHT - (UNDER_HALFWAY)
    #define BOUNDARY_TOP_LEFT WIN_DIF
    #define BOUNDARY_BOTTOM_RIGHT WIN_TOTAL - WIN_DIF
    #define PENALTY 0

    #anomaly_scores = [0] * len(R)
    # Initialize anomaly_scores as a NumPy array filled with zeros
    anomaly_scores = np.zeros((height, width), dtype=np.float32)

    # Reshape anomaly_scores to match the shape of R
    anomaly_scores = anomaly_scores.reshape((height, width))

    # Initialize window as a NumPy array filled with -1
    window = -np.ones((win_total, win_total), dtype=np.float32)

    # Initialize window_c as a NumPy array filled with zeros
    window_c = np.zeros((win_total, win_total, code), dtype=np.float32)

    # Initialize code_dist as a NumPy array filled with -1
    code_dist = -np.ones((win_total, win_total), dtype=np.float32)

    # Initialize work_mat as a NumPy array filled with zeros
    work_mat = np.zeros((win_total, win_total), dtype=np.float32)

    # Initialize weight as a NumPy array filled with zeros
    weight = np.zeros((win_total, win_total), dtype=np.float32)


    padding = dif + (size-1)//2
    Rshaped = R.reshape(height, width)
    Cshaped = C.reshape(height, width, 12)
    padded_data = np.pad(Rshaped, ((padding, padding), (padding, padding)), mode='constant', constant_values=-1)
    padded_codes = np.pad(Cshaped, ((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)

    for j in range(padding, height + 2*padding):
        for i in range(padding, width + 2*padding):
            #Get window
            start_row = i - padding
            end_row = i + padding + 1
            start_col = j - padding
            end_col = j + padding + 1
            window = padded_data[start_col:end_col, start_row:end_row]
            window_c = padded_codes[start_col:end_col, start_row:end_row, :]       

        
            if window[padding, padding] != -1:
                #count = count + 1
            
                r_mean = 0
                r_total = 0
                std_dev = 0.0
                count = 0
                anomaly_score = 0
                #boundary_t = start_col + BOUNDARY_TOP_LEFT
                #boundary_b = start_col + BOUNDARY_BOTTOM_RIGHT
                #boundary_l = start_row + BOUNDARY_TOP_LEFT
                #boundary_r = start_row + BOUNDARY_BOTTOM_RIGHT

                ### Get average
                for k in range(win_total):
                    for z in range(win_total):
                        if not (((z >= BOUNDARY_TOP_LEFT) and (z < BOUNDARY_BOTTOM_RIGHT))
                                and ((k >= BOUNDARY_TOP_LEFT) and (k < BOUNDARY_BOTTOM_RIGHT))):
                            r_total += window[k,z]
                            if window[k,z] != -1:
                                count += 1
                        else:
                            continue
                r_total = r_total + NEIGHBOURS - count
                r_mean = r_total / count
                #print("r_mean:", r_mean)

                ### Get standard deviation
                comp = 0
                # pad_mean
                for k in range(win_total):
                    for z in range(win_total):
                        if not ((z >= BOUNDARY_TOP_LEFT and z < BOUNDARY_BOTTOM_RIGHT)
                                and (k >= BOUNDARY_TOP_LEFT and k < BOUNDARY_BOTTOM_RIGHT)):
                            work_mat[k,z] = r_mean if window[k,z] == -1 else window[k,z]
                        else:
                            continue

                # square_difs
                for k in range(win_total):
                    for z in range(win_total):
                        if not ((z >= BOUNDARY_TOP_LEFT and z < BOUNDARY_BOTTOM_RIGHT)
                                and (k >= BOUNDARY_TOP_LEFT and k < BOUNDARY_BOTTOM_RIGHT)):
                            comp = work_mat[k,z] - r_mean
                            std_dev = std_dev + (comp * comp)
                        else:
                            continue

                std_dev = math.sqrt(std_dev / count)
                #print("std_dev:", std_dev)

                # Generate penalties

                for k in range(win_total):
                    for z in range(win_total):
                        if not ((z >= BOUNDARY_TOP_LEFT and z < BOUNDARY_BOTTOM_RIGHT)
                                and (k >= BOUNDARY_TOP_LEFT and k < BOUNDARY_BOTTOM_RIGHT)):
                            if abs(window[k,z] - r_mean) > std_dev:
                                weight[k,z] = 0
                            else:
                                if window[k,z] != 0:
                                    weight[k,z] = 1 / window[k,z]
                                else:
                                    weight[k,z] = 0
                        else:
                            continue

                # Generate code distance
                #print("weight:", weight)

                temp = 0

                # zero_dist
                for k in range(win_total):
                    for z in range(win_total):
                        if not ((z >= BOUNDARY_TOP_LEFT and z < BOUNDARY_BOTTOM_RIGHT)
                                and (k >= BOUNDARY_TOP_LEFT and k < BOUNDARY_BOTTOM_RIGHT)):
                            code_dist[k,z] = 0
                        else:
                            continue

                # code_distance
                for k in range(win_total):
                    for z in range(win_total):
                        if not ((z >= BOUNDARY_TOP_LEFT and z < BOUNDARY_BOTTOM_RIGHT)
                                and (k >= BOUNDARY_TOP_LEFT and k < BOUNDARY_BOTTOM_RIGHT)):
                            for c in range(code):
                                temp = window_c[k,z,c] - window_c[under_halfway,under_halfway,c]
                                code_dist[k,z] += temp * temp

                            code_dist[k,z] = math.sqrt(code_dist[k,z])
                        else:
                            continue

                #print("code_dist:", code_dist)

                # anomaly_score
                for k in range(win_total):
                    for z in range(win_total):
                        if not ((z >= BOUNDARY_TOP_LEFT and z < BOUNDARY_BOTTOM_RIGHT)
                                and (k >= BOUNDARY_TOP_LEFT and k < BOUNDARY_BOTTOM_RIGHT)):
                            anomaly_score += code_dist[k,z] * weight[k,z]
                        else:
                            continue

                anomaly_score = window[under_halfway,under_halfway]*anomaly_score/count
                #print("anomaly_score:", anomaly_score)
                #print("Indexes are :", i, j)
                #anomaly_scores = np.append(anomaly_scores, anomaly_score)     
                anomaly_scores[j-padding, i-padding] = anomaly_score           
            else: 
                continue

    return anomaly_scores


# Example usage:
# con = {'height': 12, 'width': 653904, 'penalty': 0.0, 'dif': 5, 'size': 2, 'code': 1}
# R = np.random.rand(12, 653904)
# C = np.random.rand(12, 653904, 1)
# result = new_aw_window_skip(con, R, C)
def opt_novel_dsw(con, R, C):
   
    height = con['height']
    width = con['width']
    dif = con['win_dif']
    size = con['win_size']
    code = con['code_size']

    win_total = size + 2 * dif # height and width if inner window.
    neighbours = (win_total * win_total - size * size)
    under_halfway = dif + (size - 1) // 2
    BOUNDARY_BOTTOM_RIGHT = dif
    BOUNDARY_TOP_LEFT = win_total - dif
    WIN_LENGTH = win_total
    NEIGHBOURS = ( (win_total)*(win_total) - (size*size) )

    #define UNDER_HALFWAY (WIN_DIF + (WIN_SIZE - 1)/2)
    #define OVER_HALFWAY (WIN_SIZE + (WIN_DIF * 2) - UNDER_HALFWAY)
    #define WIN_TOTAL (WIN_LENGTH)
    #define VALID_TOP HEIGHT - (UNDER_HALFWAY)
    #define BOUNDARY_TOP_LEFT WIN_DIF
    #define BOUNDARY_BOTTOM_RIGHT WIN_TOTAL - WIN_DIF
    #define PENALTY 0

    #anomaly_scores = [0] * len(R)
    # Initialize anomaly_scores as a NumPy array filled with zeros
    anomaly_scores = np.zeros((height, width), dtype=np.float32)

    # Reshape anomaly_scores to match the shape of R
    anomaly_scores = anomaly_scores.reshape((height, width))

    # Initialize window as a NumPy array filled with -1
    window = -np.ones((win_total, win_total), dtype=np.float32)

    # Initialize window_c as a NumPy array filled with zeros
    window_c = np.zeros((win_total, win_total, code), dtype=np.float32)

    # Initialize code_dist as a NumPy array filled with -1
    code_dist = -np.ones((win_total, win_total), dtype=np.float32)


    # Initialize weight as a NumPy array filled with zeros
    weight = np.zeros((win_total, win_total), dtype=np.float32)
    scores_count = 0
    include_slice1 = slice(0, dif)
    include_slice2 = slice(win_total - dif, win_total)


            # Create a mask for the inner window to exclude it from subsequent calculations
    window_mask = np.zeros_like(window, dtype=bool)
    window_mask[include_slice1, :] = True
    window_mask[include_slice2, :] = True
    window_mask[:, include_slice1] = True
    window_mask[:, include_slice2] = True


    padding = dif + (size-1)//2
    Rshaped = R.reshape(height, width)
    Cshaped = C.reshape(height, width, 12)
    padded_data = np.pad(Rshaped, ((padding, padding), (padding, padding)), mode='constant', constant_values=-1)
    padded_codes = np.pad(Cshaped, ((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)
    for j in range(padding, height + padding):
        for i in range(padding, width + padding):
            #Get window
            first_row = j - padding
            last_row = j + padding + 1
            first_col = i - padding
            last_col = i + padding + 1
            
            window = padded_data[first_row:last_row, first_col:last_col]
            window_c = padded_codes[first_row:last_row, first_col:last_col, :]       



            if window[padding, padding] != -1:
                r_mean = 0
                r_total = 0
                std_dev = 0.0
                count = 0
                anomaly_score = 0

                mid_val = window[padding, padding]

                r_total = np.sum(np.where(window_mask, window, 0))
                count = np.sum(np.logical_and(window_mask, window != -1))
                r_total += (NEIGHBOURS - count)
                r_mean = r_total / count
                
                
                comp_mat = window.copy() 
                work_mat = window.copy()

                work_mat[work_mat == -1] = r_mean
                comp = work_mat - r_mean
                std_dev = math.sqrt(np.sum(np.where(window_mask, comp ** 2, 0)) / count)

                #Penalties
                weight = np.where(np.abs(comp_mat - r_mean) > std_dev, 0, 1 / comp_mat)
                weight[comp_mat == -1] = 0
                #Code distance
                temp = window_c - window_c[under_halfway, under_halfway, np.newaxis, :]
                code_dist = np.sqrt(np.sum(temp ** 2, axis=2))

                # Calculate code_dist only for the region covered by window_mask
                #code_dist = np.sqrt(np.sum(temp[window_mask, :] ** 2, axis=1))

                # Anomaly score
                anomaly_score = np.sum(code_dist[window_mask] * weight[window_mask]) * mid_val / count

                anomaly_scores[j-padding, i-padding] = anomaly_score           
                """
                if(j == height + padding - 1):
                    print("work one")
                    print(work_mat)
                    print("comp one")
                    print(comp_mat)
                    print("Interesting values:")
                    print("score :", anomaly_score)
                    print("code_dist:", code_dist)
                    print("temp:", temp)
                    print("deviation:",std_dev)
                """


            else: 
                continue
    return anomaly_scores


# Example usage:
# con = {'height': 12, 'width': 653904, 'penalty': 0.0, 'dif': 5, 'size': 2, 'code': 1}
# R = np.random.rand(12, 653904)
# C = np.random.rand(12, 653904, 1)
# result = new_aw_window_skip(con, R, C)
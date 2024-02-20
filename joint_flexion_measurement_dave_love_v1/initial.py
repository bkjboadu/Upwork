def plot_angle_graph_and_save(frames, angles, save_as, release_frame):
    x = np.array(frames)
    y = np.array(angles)

    def normalize_to_range(arrays, target_range=(-10, 30)):
        # Find the min and max values across all arrays
        min_val = np.min(arrays)
        max_val = np.max(arrays)

        # Calculate the scaling factor
        scale_factor = (target_range[1] - target_range[0]) / (max_val - min_val)

        # Normalize each array
        normalized_arrays = [(array - min_val) * scale_factor + target_range[0] for array in arrays]

        return normalized_arrays, min_val, scale_factor

    labels = ["wrist_angle", "elbow_angle", "shoulder_angle", "hip_angle", "knee_angle", "ankle_angle"]
    labels_to_normalize = ["elbow_angle", "hip_angle", "knee_angle", "ankle_angle"]
    normalization_threshold = {"hip_angle": (-30, 0), "knee_angle": (0, -45), "ankle_angle": (-20, 40)}
    df = pd.DataFrame(y, index=x, columns=labels)

    labels_to_plot = ["elbow_angle", "shoulder_angle", "wrist_angle", "hip_angle", "knee_angle", "ankle_angle"]

    for col in labels_to_plot:
        y = df.loc[:, col]

        if col in labels_to_plot[3:]:
            y, min_val, scale_factor = normalize_to_range(y)
        else:
            y, min_val, scale_factor = y.values, 0, 1

        x_smooth, y_smooth = kalman_filter(x, y)
        
        # Adjust the initial state of the Kalman filter based on normalization
        kf.x[0] = min_val * scale_factor

        if col == 'wrist_angle':
            y_smooth = [-value for value in y_smooth]

        if col in labels_to_plot[3:]:
            plt.plot(x_smooth, y_smooth, label=col, linewidth=0.4)
        else:
            plt.plot(x_smooth, y_smooth, label=col)

    plt.axvline(x=release_frame, color='red', linestyle='--', label='Released')
    plt.title("Joint Flexion graph")
    plt.xlabel("frames")
    plt.ylabel("angles")
    plt.legend(loc="upper left", prop={'size': 5})

    # Save the plot
    plt.savefig(save_as)
    plt.close()

def plot_angle_graph_and_save(frames,angles,save_as,release_frame):
    x = np.array(frames)
    y = np.array(angles)

    def normalize_to_range(arrays, target_range=(-10, 30)):
        # Find the min and max values across all arrays
        min_val = np.min(arrays)
        max_val = np.max(arrays)

        # Calculate the scaling factor
        scale_factor = (target_range[1] - target_range[0]) / (max_val - min_val)

        # Normalize each array
        normalized_arrays = [(array - min_val) * scale_factor + target_range[0] for array in arrays]

        return normalized_arrays,min_val,scale_factor


    labels = ["wrist_angle", "elbow_angle", "shoulder_angle", "hip_angle", "knee_angle", "ankle_angle"]
    labels_to_normalize = ["elbow_angle","hip_angle", "knee_angle", "ankle_angle"]
    normalization_threshold = { "hip_angle":(-30,0), "knee_angle":(0,-45), "ankle_angle":(-20,40)}
    df = pd.DataFrame(y,index=x,columns=labels)
    


    labels_to_plot = ["elbow_angle", "shoulder_angle", "wrist_angle","hip_angle", "knee_angle", "ankle_angle"]

    for col in labels_to_plot:
        y = df.loc[:,col]
        if col in labels_to_plot[3:]:
            y = normalize_to_range(y)
        x_smooth,y_smooth = kalman_filter(x,y)
        if col == 'wrist_angle':  
            y_smooth = [-value for value in y_smooth]

        if col in labels_to_plot[3:]:
            plt.plot(x_smooth, y_smooth, label=col,linewidth=0.4)
        else:
            plt.plot(x_smooth, y_smooth, label=col)

    plt.axvline(x=release_frame, color='red', linestyle='--', label='Released')
    plt.title("Joint Flexion graph")
    plt.xlabel("frames")
    plt.ylabel("angles")
    plt.legend(loc="upper left", prop={'size':5})

    # Save the plot
    plt.savefig(save_as)
    plt.close()
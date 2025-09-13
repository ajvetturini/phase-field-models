from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import numpy as np
from scipy.ndimage import gaussian_filter

def _infer_N(filepath: str) -> int:
    """ Infers the number of rows per frame of a trajectory file. """
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("# step ="):  # Detect first metadata line
                break

        # Read and count the number of numerical lines until the next comment line
        N = 0
        for line in file:
            if line.startswith("# step ="):  # Stop counting when next comment is reached
                break
            N += 1

    return N

def _read_in_filepath(filepath, sigma: float, interpolation_factor: int, smoothen_animation: bool):
    all_frames = []
    frame_data = []
    with open(filepath, 'r') as input_file:
        for line in input_file:
            cleaned_line = line.split()
            if not cleaned_line:
                continue
            elif '#' in line or cleaned_line[1] == 'step':
                if frame_data:
                    all_frames.append(np.array(frame_data))
                frame_data = []  # reset
                continue
            frame_data.extend(np.array([float(val) for val in cleaned_line]))

    # Perform smoothening if desired:
    smoothed_frames = [gaussian_filter(frame, sigma=sigma) if sigma > 0 else frame for frame in all_frames]

    interpolated_frames = []
    N = int(np.sqrt(all_frames[0].shape[0]))
    for i in range(len(smoothed_frames) - 1):
        frame1 = smoothed_frames[i]
        frame2 = smoothed_frames[i + 1]
        interpolated_frames.append(frame1.reshape(N, N))
        for j in range(1, interpolation_factor):
            alpha = j / interpolation_factor
            interpolated_frame = (1 - alpha) * frame1 + alpha * frame2
            interpolated_frames.append(interpolated_frame.reshape(N, N))

    interpolated_frames.append(smoothed_frames[-1].reshape(N, N))

    if not interpolated_frames or not smoothen_animation:
        fixed_frames = []
        for f in all_frames:
            if len(f.shape) == 1:
                f = f.reshape(N, N)
                fixed_frames.append(f)
            else:
                fixed_frames = all_frames
                break
        frames_to_return = fixed_frames
    else:
        frames_to_return = interpolated_frames
    return frames_to_return


def animate(filepath: str, smoothen_animation: bool = False, interpolation_factor: int = 5,
            sigma: float = 0.0, **kwargs):
    """ Creates and saves an animation from a data file with the specified format """
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.15)

    all_frames = _read_in_filepath(filepath, sigma, interpolation_factor, smoothen_animation)
    norm = colors.Normalize(vmin=np.min(all_frames), vmax=np.max(all_frames))
    image = ax.imshow(all_frames[0], norm=norm, cmap=kwargs.get('cmap', 'plasma'))
    cbar = fig.colorbar(image, label="$\psi$")
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])

    slider = widgets.Slider(ax_slider, 'Frame', 0, len(all_frames) - 1, valinit=0, valstep=1)
    replay_button = widgets.Button(plt.axes([0.01, 0.03, 0.1, 0.05]), 'Replay')
    pause_button = widgets.Button(plt.axes([0.01, 0.03 + 0.05 + 0.01, 0.1, 0.05]), 'Pause')
    start_button = widgets.Button(plt.axes([0.01, 0.03 + 2 * (0.05 + 0.01), 0.1, 0.05]), 'Start')

    def update_figure(j):
        image.set_data(all_frames[j])
        return [image]

    slider.on_changed(update_figure)

    def animate_func(j):
        slider.set_val(j)
        return [image]

    interval = kwargs.get('interval', 200)
    anim = animation.FuncAnimation(fig, animate_func, frames=len(all_frames),
                                   interval=interval // (interpolation_factor + 1) if interpolation_factor >= 0 else interval,
                                   blit=True, repeat=True, repeat_delay=500)

    def replay(event):
        slider.set_val(0)
        anim.frame_seq = anim.new_frame_seq()
        anim.resume()
    replay_button.on_clicked(replay)

    def pause(event):
        anim.pause()
    pause_button.on_clicked(pause)

    def start(event):
        anim.resume()
    start_button.on_clicked(start)

    if kwargs.get('export', False):
        try:
            anim.save(filepath + ".mp4", writer='ffmpeg')
            print(f"Animation saved as {filepath}.mp4")
        except ImportError:
            print('FFMPEG not installed, cannot output .mp4...')
        except Exception as e:
            print(f'Error saving .mp4: {e}')

        try:
            anim.save(filepath + ".gif", writer='pillow')
            print(f"Animation saved as {filepath}.gif")
        except ImportError:
            print('Pillow not installed, cannot output .gif...')
        except Exception as e:
            print(f'Error saving .gif: {e}')

    plt.show()

def animate_multispecies(list_of_filepaths: list, smoothen_animation: bool = False, interpolation_factor: int = 5,
                         sigma: float = 0.0, export_video_filepath_no_extension: str = None, **kwargs):
    """ Animates a multi-species phase field model """
    all_species = []
    for fp in list_of_filepaths:
        all_frames = _read_in_filepath(fp, sigma, interpolation_factor, smoothen_animation)
        all_species.append(all_frames)

    expected_len = len(all_species[0])
    assert all(len(sub) == expected_len for sub in all_species), "ERROR: All trajectories do not have the same length..."
    num_species = len(all_species)

    per_frame_data = []
    for i in range(expected_len):
        frame_i = []
        rho_max = 0
        for j in range(num_species):
            conf_data = all_species[j][i]
            frame_i.append(conf_data)
            data_max = np.max(conf_data)
            if data_max > rho_max:
                rho_max = data_max

        # After reading in all species at frame i, convert data;
        final_frame_data = (np.dstack(frame_i / rho_max) * 259.999).astype(np.uint8)  # Convert to b256 safely
        per_frame_data.append(final_frame_data)

    # Once the per_frame_data is constructed, we can populate the animation and show:
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.15)
    #norm = colors.Normalize(vmin=np.min(per_frame_data), vmax=np.max(per_frame_data))
    #image = ax.imshow(per_frame_data[0], norm=norm, **kwargs)
    image = ax.imshow(per_frame_data[0], **kwargs)
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])

    slider = widgets.Slider(ax_slider, 'Frame', 0, len(per_frame_data) - 1, valinit=0, valstep=1)
    replay_button = widgets.Button(plt.axes([0.01, 0.03, 0.1, 0.05]), 'Replay')
    pause_button = widgets.Button(plt.axes([0.01, 0.03 + 0.05 + 0.01, 0.1, 0.05]), 'Pause')
    start_button = widgets.Button(plt.axes([0.01, 0.03 + 2 * (0.05 + 0.01), 0.1, 0.05]), 'Start')

    def update_figure(_j):
        image.set_data(per_frame_data[_j])
        return [image]

    slider.on_changed(update_figure)

    def animate_func(_j):
        slider.set_val(_j)
        return [image]

    interval = kwargs.get('interval', 200)
    anim = animation.FuncAnimation(fig, animate_func, frames=len(per_frame_data),
                                   interval=interval // (
                                               interpolation_factor + 1) if interpolation_factor >= 0 else interval,
                                   blit=True, repeat=True, repeat_delay=500)

    def replay(event):
        slider.set_val(0)
        anim.frame_seq = anim.new_frame_seq()
        anim.resume()

    replay_button.on_clicked(replay)

    def pause(event):
        anim.pause()

    pause_button.on_clicked(pause)

    def start(event):
        anim.resume()

    start_button.on_clicked(start)

    if export_video_filepath_no_extension is not None:
        try:
            anim.save(export_video_filepath_no_extension + ".mp4", writer='ffmpeg')
            print(f"Animation saved as {export_video_filepath_no_extension}.mp4")
        except ImportError:
            print('FFMPEG not installed, cannot output .mp4...')
        except Exception as e:
            print(f'Error saving .mp4: {e}')

        try:
            anim.save(export_video_filepath_no_extension + ".gif", writer='pillow')
            print(f"Animation saved as {export_video_filepath_no_extension}.gif")
        except ImportError:
            print('Pillow not installed, cannot output .gif...')
        except Exception as e:
            print(f'Error saving .gif: {e}')

    plt.show()

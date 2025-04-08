from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import numpy as np
from scipy.ndimage import gaussian_filter
import re  # Import the regular expression module

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

'''def animate(filepath: str, **kwargs):
    """
    Creates and saves an animation from a data file.

    Args:
        filepath (str): Path to the input file containing numerical data.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.15)  # Adjust layout to fit UI elements
    N = _infer_N(filepath)

    frames = []
    with open(filepath) as input_file:
        def list_chunks(lines, n):
            """Splits lines into chunks of size n."""
            for i in range(0, len(lines), n):
                yield lines[i:i + n]

        for lines in list_chunks(input_file.readlines(), N+1):
            data = np.loadtxt(lines[1:])
            frames.append(data)

    # Normalize colors based on the final frame
    norm = colors.Normalize(vmin=frames[-1].min(), vmax=frames[-1].max())
    image = plt.imshow(frames[0], norm=norm)
    cbar = fig.colorbar(image, label="$\psi$")
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])

    # Create widgets for user
    slider = widgets.Slider(ax_slider, 'Frame', 0, len(frames) - 1, valinit=0, valstep=1)
    replay_button = widgets.Button(plt.axes([0.01, 0.03, 0.1, 0.05]), 'Replay')
    pause_button = widgets.Button(plt.axes([0.01, 0.03 + 0.05 + 0.01, 0.1, 0.05]), 'Pause')
    start_button = widgets.Button(plt.axes([0.01, 0.03 + 2 * (0.05 + 0.01), 0.1, 0.05]), 'Start')

    def update_figure(j):
        """Updates the figure for animation."""
        image.set_data(frames[j])
        return [image]

    slider.on_changed(update_figure)

    def animate_func(j):
        slider.set_val(j)  # Move slider along with animation
        return [image]

    # Create animation
    interval = kwargs.get('interval', 200)
    anim = animation.FuncAnimation(fig, animate_func, frames=len(frames), interval=interval,
                                   blit=True, repeat=True, repeat_delay=500)

    # Add buttons
    def replay(event):
        slider.set_val(0)  # Reset the slider to the first frame
        anim.frame_seq = anim.new_frame_seq()  # Reset the frame sequence
        anim.resume()
    replay_button.on_clicked(replay)

    def pause(event):
        anim.pause()
    pause_button.on_clicked(pause)

    def start(event):
        anim.resume()
    start_button.on_clicked(start)

    # Save animation
    try:
        anim.save(filepath + ".mp4")  # Requires ffmpeg
    except:
        print('FFMPEG not installed, can not output .mp4...')

    anim.save(filepath + ".gif")  # Saves as a GIF

    plt.show()'''

def animate(filepath: str, interpolation_factor: int = 5, sigma: float = 0.0, **kwargs):
    """
    Creates and saves an animation from a data file with the specified format.

    Args:
        filepath (str): Path to the input file containing numerical data.
                        The file should have frames separated by lines starting with '# step',
                        followed by a line indicating 'size = NxN', and then N lines
                        each containing N space-separated floating-point numbers.
        interpolation_factor (int): Number of interpolated frames to create between
                                     each pair of original frames.
        sigma (float): Standard deviation for Gaussian smoothing applied to each frame.
                       If 0.0, no smoothing is applied.
        **kwargs: Additional keyword arguments passed to matplotlib.animation.FuncAnimation
                  and matplotlib.axes.Axes.imshow.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.15)

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
            try:
                frame_data.extend(np.array([float(val) for val in cleaned_line]))
            except:
                print(f'')
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

    if not interpolated_frames:
        print("No frames to animate after processing. Aborting.")
        return

    norm = colors.Normalize(vmin=np.min(interpolated_frames), vmax=np.max(interpolated_frames))
    image = ax.imshow(interpolated_frames[0], norm=norm)
    cbar = fig.colorbar(image, label="$\psi$")
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])

    slider = widgets.Slider(ax_slider, 'Frame', 0, len(interpolated_frames) - 1, valinit=0, valstep=1)
    replay_button = widgets.Button(plt.axes([0.01, 0.03, 0.1, 0.05]), 'Replay')
    pause_button = widgets.Button(plt.axes([0.01, 0.03 + 0.05 + 0.01, 0.1, 0.05]), 'Pause')
    start_button = widgets.Button(plt.axes([0.01, 0.03 + 2 * (0.05 + 0.01), 0.1, 0.05]), 'Start')

    def update_figure(j):
        image.set_data(interpolated_frames[j])
        return [image]

    slider.on_changed(update_figure)

    def animate_func(j):
        slider.set_val(j)
        return [image]

    interval = kwargs.get('interval', 200)
    anim = animation.FuncAnimation(fig, animate_func, frames=len(interpolated_frames),
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
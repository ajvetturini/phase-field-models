from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import numpy as np

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

def animate(filepath: str, **kwargs):
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

        for lines in list_chunks(input_file.readlines(), N):
            data = np.loadtxt(lines)
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

    plt.show()

#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Figure with horizontal scrollbar and play capabilities

For latest version, go to https://github.com/bilylee/videofig

Basic usage
-----------
Creates a figure with a horizontal scrollbar and shortcuts to scroll automatically.
The scroll range is 0 to NUM_FRAMES - 1. The function REDRAW_FUN(F, AXES) is called to
redraw at scroll position F (for example, REDRAW_FUNC can show the frame F of a video)
using AXES for drawing. F is an integer, AXES is a instance of [Axes class](https://matplotlib.org/api/axes_api.html)

This can be used not only to play and analyze standard videos, but it also lets you place
any custom Matplotlib plots and graphics on top.

The keyboard shortcuts are:
  Enter(Return) -- play/pause video (25 frames-per-second default).
  Backspace -- play/pause video 5 times slower.
  Right/left arrow keys -- advance/go back one frame.
  Page down/page up -- advance/go back 30 frames.
  Home/end -- go to first/last frame of video.

Advanced usage
--------------
videofig(NUM_FRAMES, REDRAW_FUNC, FPS, BIG_SCROLL)
Also specifies the speed of the play function (frames-per-second) and
the frame step of page up/page down (or empty for defaults).

videofig(NUM_FRAMES, REDRAW_FUNC, FPS, BIG_SCROLL, KEY_FUNC)
Also calls KEY_FUNC(KEY) with any keys that weren't processed, so you
can add more shortcut keys (or empty for none).

Example 1: Plot a dynamic sine wave
---------
  import numpy as np

  def redraw_fn(f, axes):
    amp = float(f) / 3000
    f0 = 3
    t = np.arange(0.0, 1.0, 0.001)
    s = amp * np.sin(2 * np.pi * f0 * t)
    if not redraw_fn.initialized:
      redraw_fn.l, = axes.plot(t, s, lw=2, color='red')
      redraw_fn.initialized = True
    else:
      redraw_fn.l.set_ydata(s)

  redraw_fn.initialized = False

  videofig(100, redraw_fn)

Example 2: Show images in a custom directory
---------
  import os
  import glob
  from scipy.misc import imread

  img_dir = 'YOUR-IMAGE-DIRECTORY'
  img_files = glob.glob(os.path.join(video_dir, '*.jpg'))

  def redraw_fn(f, axes):
    img_file = img_files[f]
    img = imread(img_file)
    if not redraw_fn.initialized:
      redraw_fn.im = axes.imshow(img, animated=True)
      redraw_fn.initialized = True
    else:
      redraw_fn.im.set_array(img)
  redraw_fn.initialized = False

  videofig(len(img_files), redraw_fn, play_fps=30)

Example 3: Show images together with object bounding boxes
----------
  import os
  import glob
  from scipy.misc import imread
  from matplotlib.pyplot import Rectangle

  video_dir = 'YOUR-VIDEO-DIRECTORY'

  img_files = glob.glob(os.path.join(video_dir, '*.jpg'))
  box_files = glob.glob(os.path.join(video_dir, '*.txt'))

  def redraw_fn(f, axes):
    img = imread(img_files[f])
    box = bbread(box_files[f])  # Define your own bounding box reading utility
    x, y, w, h = box
    if not redraw_fn.initialized:
      im = axes.imshow(img, animated=True)
      bb = Rectangle((x, y), w, h,
                     fill=False,  # remove background
                     edgecolor="red")
      axes.add_patch(bb)
      redraw_fn.im = im
      redraw_fn.bb = bb
      redraw_fn.initialized = True
    else:
      redraw_fn.im.set_array(img)
      redraw_fn.bb.set_xy((x, y))
      redraw_fn.bb.set_width(w)
      redraw_fn.bb.set_height(h)
  redraw_fn.initialized = False

  videofig(len(img_files), redraw_fn, play_fps=30)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider


def videofig(num_frames, redraw_func, play_fps=25, big_scroll=30, key_func=None,
             grid_specs=None, layout_specs=None, figname=None, *args):
  """Figure with horizontal scrollbar and play capabilities

  This script is mainly inspired by the elegant work of João Filipe Henriques
    https://www.mathworks.com/matlabcentral/fileexchange/29544-figure-to-play-and-analyze-videos-with-custom-plots-on-top?focused=5172704&tab=function

  :param num_frames: an integer, number of frames in a sequence
  :param redraw_func: callable with signature redraw_func(f, axes)
                      used to draw a new frame at position f using axes, which is a instance of Axes class in matplotlib 
  :param play_fps: an integer, number of frames per second, used to control the play speed
  :param big_scroll: an integer, big scroll number used when pressed page down or page up keys. 
  :param key_func: optional callable which signature key_func(key), used to provide custom key shortcuts.
  :param grid_specs: optional dictionary, used to specify the gridspec of the main drawing pane. 
                     For example, grid_specs = {'nrows': 2, 'ncols': 2} will create a gridspec with 2 rows and 2 cols.
  :param layout_specs: optional list, used to specify the layout of the gridspec of the main drawing pane.
                     For example, layout_specs = ['[:, 0]', '[:, 1]'] means:
                        gs = ... Some code to create the main drawing pane gridspec ...
                        ax1 = plt.subplot(gs[:, 0])
                        ax2 = plt.subplot(gs[:, 1])
  :param args: other optional arguments
  :return: None
  """
  # Check arguments
  check_int_scalar(num_frames, 'num_frames')
  check_callback(redraw_func, 'redraw_func')
  check_int_scalar(play_fps, 'play_fps')
  check_int_scalar(big_scroll, 'big_scroll')
  if key_func:
    check_callback(key_func, 'key_func')

  # Initialize figure
  if not figname:
    figname = 'VideoPlayer'
  fig_handle = plt.figure(num=figname)

  # We use GridSpec to support embedding multiple plots in the main drawing pane.
  # A nested grid demo can be found at https://matplotlib.org/users/plotting/examples/demo_gridspec06.py
  # Construct outer grid, which contains the main drawing pane and slider ball
  outer_grid = gridspec.GridSpec(2, 1,
                                 left=0, bottom=0, right=1, top=1,
                                 height_ratios=[97, 3], wspace=0.0, hspace=0.0)

  # Construct inner grid in main drawing pane
  if grid_specs is None:
    grid_specs = {'nrows': 1, 'ncols': 1}  # We will have only one axes by default

  inner_grid = gridspec.GridSpecFromSubplotSpec(subplot_spec=outer_grid[0], **grid_specs)
  if layout_specs:
    # eval() can't work properly in list comprehension which is inside a function.
    # Refer to: http://bugs.python.org/issue5242
    # Maybe I should find another way to implement layout_specs without using eval()...
    axes_handle = []
    for spec in layout_specs:
      axes_handle.append(plt.Subplot(fig_handle, eval('inner_grid' + spec)))
  else:
    num_inner_plots = grid_specs['nrows'] * grid_specs['ncols']
    axes_handle = [plt.Subplot(fig_handle, inner_grid[i]) for i in range(num_inner_plots)]

  for ax in axes_handle: fig_handle.add_subplot(ax)

  if len(axes_handle) == 1:
    axes_handle = axes_handle[0]
    axes_handle.set_axis_off()

  # Build scrollbar
  scroll_axes_handle = plt.Subplot(fig_handle, outer_grid[1])
  scroll_axes_handle.set_facecolor('lightgoldenrodyellow')
  fig_handle.add_subplot(scroll_axes_handle)
  scroll_handle = Slider(scroll_axes_handle, '', 0.0, num_frames - 1, valinit=0.0)

  def draw_new(_):
    redraw_func(int(scroll_handle.val), axes_handle)
    fig_handle.canvas.draw_idle()

  def scroll(new_f):
    new_f = min(max(new_f, 0), num_frames - 1)  # clip in the range of [0, num_frames - 1]
    cur_f = scroll_handle.val

    # Stop player at the end of the sequence
    if new_f == (num_frames - 1):
      play.running = False

    if cur_f != new_f:
      # move scroll bar to new position
      scroll_handle.set_val(new_f)

    return axes_handle

  def play(period):
    play.running ^= True  # Toggle state
    if play.running:
      frame_idxs = range(int(scroll_handle.val), num_frames)
      play.anim = FuncAnimation(fig_handle, scroll, frame_idxs,
                                interval=1000 * period, repeat=False)
      plt.draw()
    else:
      play.anim.event_source.stop()

  # Set initial player state
  play.running = False

  def key_press(event):
    key = event.key
    f = scroll_handle.val
    if key == 'left':
      scroll(f - 1)
    elif key == 'right':
      scroll(f + 1)
    elif key == 'pageup':
      scroll(f - big_scroll)
    elif key == 'pagedown':
      scroll(f + big_scroll)
    elif key == 'home':
      scroll(0)
    elif key == 'end':
      scroll(num_frames - 1)
    elif key == 'enter':
      play(1 / play_fps)
    elif key == 'backspace':
      play(5 / play_fps)
    else:
      if key_func:
        key_func(key)

  # Register events
  scroll_handle.on_changed(draw_new)
  fig_handle.canvas.mpl_connect('key_press_event', key_press)

  # Draw initial frame
  redraw_func(0, axes_handle)

  # Start playing
  play(1 / play_fps)

  # plt.show() has to be put in the end of the function,
  # otherwise, the program simply won't work, weird...
  plt.show()


def check_int_scalar(a, name):
  assert isinstance(a, int), '{} must be a int scalar, instead of {}'.format(name, type(name))


def check_callback(a, name):
  # Check http://stackoverflow.com/questions/624926/how-to-detect-whether-a-python-variable-is-a-function
  # for more details about python function type detection.
  assert callable(a), '{} must be callable, instead of {}'.format(name, type(name))


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--imgdir", type=str, default=None, help="directory containing JPEG images")
  args = parser.parse_args()

  if args.imgdir:
    import os
    import glob
    from scipy.misc import imread

    img_files = sorted(glob.glob(os.path.join(args.imgdir, '*.jpg')))


    def redraw_fn(f, axes):
      img_file = img_files[f]
      img = imread(img_file)
      if not redraw_fn.initialized:
        redraw_fn.im = axes.imshow(img, animated=True)
        redraw_fn.initialized = True
      else:
        redraw_fn.im.set_array(img)


    redraw_fn.initialized = False
    videofig(len(img_files), redraw_fn, play_fps=30)

  else:
    import numpy as np


    def redraw_fn(f, axes):
      amp = float(f) / 3000
      f0 = 3
      t = np.arange(0.0, 1.0, 0.001)
      s = amp * np.sin(2 * np.pi * f0 * t)
      if not redraw_fn.initialized:
        redraw_fn.l, = axes.plot(t, s, lw=2, color='red')
        redraw_fn.initialized = True
      else:
        redraw_fn.l.set_ydata(s)


    redraw_fn.initialized = False
    videofig(100, redraw_fn)

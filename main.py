# Copyright 2026 Abhishek Mishra <abhishekdoo@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Configuration ---
cap = cv2.VideoCapture(0)
SCALE_FACTOR = 0.1  # Reduce resolution for performance (0.1 = 10% size)
SIGMOID_STEEPEN = 0.1 # Adjusts how sharp the transition is

def sigmoid(x):
    # Centering around 127 (mid-gray) and scaling
    return 1 / (1 + np.exp(-SIGMOID_STEEPEN * (x - 127)))

# Initialize Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ret, img = cap.read()
    if not ret:
        return
    
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize for performance
    small_gray = cv2.resize(gray, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    h, w = small_gray.shape
    
    # 3. Apply Sigmoid
    step_size = 0.1
    z_values = np.round(sigmoid(small_gray) / step_size) * step_size

    # 4. Create Coordinate Grid
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    X, Y = np.meshgrid(x, y)
    
    # 5. Update 3D Plot
    ax.clear()
    # We flip Y to match image coordinates (top-left is 0,0)
    surf = ax.plot_surface(X, Y, z_values, cmap='viridis', edgecolor='none')
    
    ax.set_zlim(0, 1)
    ax.set_title("Webcam Pixel Intensity (Sigmoid)")
    ax.set_zlabel("Sigmoid Value")

# Run Animation
ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.show()

cap.release()
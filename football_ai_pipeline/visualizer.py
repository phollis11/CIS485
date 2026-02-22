import cv2
import numpy as np

class PitchVisualizer:
    def __init__(self, pitch_width=105, pitch_height=68, scale=5):
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.scale = scale
        self.scaled_width = int(pitch_width * scale)
        self.scaled_height = int(pitch_height * scale)

    def draw_pitch_2d(self):
        # Create a green image
        pitch = np.zeros((self.scaled_height, self.scaled_width, 3), dtype=np.uint8)
        pitch[:] = [0, 128, 0] # Green
        
        # Draw lines (simplified)
        cv2.rectangle(pitch, (0, 0), (self.scaled_width, self.scaled_height), (255, 255, 255), 2)
        cv2.line(pitch, (self.scaled_width // 2, 0), (self.scaled_width // 2, self.scaled_height), (255, 255, 255), 2)
        cv2.circle(pitch, (self.scaled_width // 2, self.scaled_height // 2), int(9.15 * self.scale), (255, 255, 255), 2)
        
        return pitch

    def draw_players_on_pitch(self, pitch, players_coords, colors):
        for coord, color in zip(players_coords, colors):
            x, y = int(coord[0] * self.scale), int(coord[1] * self.scale)
            if 0 <= x < self.scaled_width and 0 <= y < self.scaled_height:
                cv2.circle(pitch, (x, y), 5, color, -1)
        return pitch

    def calculate_pitch_control(self, team_1_xy, team_2_xy, team_1_color, team_2_color, opacity=0.5):
        """
        Implements the smooth Voronoi-based pitch control logic from the notebook.
        """
        pitch = self.draw_pitch_2d()
        voronoi = np.zeros_like(pitch)
        
        padding = 0
        y_coordinates, x_coordinates = np.indices((self.scaled_height, self.scaled_width))

        def calculate_distances(xy, x_coords, y_coords):
            if len(xy) == 0:
                return np.full_like(x_coords, 1e6, dtype=np.float32)
            # xy is in pitch meters
            # x_coords/y_coords are in pixels
            dist = np.sqrt((xy[:, 0][:, None, None] * self.scale - x_coords) ** 2 +
                           (xy[:, 1][:, None, None] * self.scale - y_coords) ** 2)
            return np.min(dist, axis=0)

        min_dist_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
        min_dist_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

        # Smooth blend logic
        steepness = 15
        distance_ratio = min_dist_2 / np.clip(min_dist_1 + min_dist_2, a_min=1e-5, a_max=None)
        blend_factor = np.tanh((distance_ratio - 0.5) * steepness) * 0.5 + 0.5

        for c in range(3):
            # team_1_color is (B, G, R)
            voronoi[:, :, c] = (blend_factor * team_1_color[c] +
                                (1 - blend_factor) * team_2_color[c]).astype(np.uint8)

        overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)
        return overlay

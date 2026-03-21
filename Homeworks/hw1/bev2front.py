import cv2
import numpy as np

class Projection(object):
    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """
        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape
        self.points = points

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """
        # 1. hardcode two camera positions and orientations in the world space
        cam_bev = [0.0, 2.5, 0.0]
        cam_front = [0.0, 1.0, 0.0]

        # calcaulte parameters
        center_x = self.width / 2
        center_y = self.height / 2
        fov_rad = fov / 180 * np.pi
        f = (self.width / 2) / np.tan(fov_rad / 2) # similar triangles, knowing that tan = width/2 / focal_length
        
        new_pixels = []

        # 2. project the points to the floor 
        for p in self.points:
            # first, we assume that the camera is looking straight down
            # then this is a 3d vector starting from the pixel on the sensor plane and pointing towards the lens center
            from_pixel_to_pinhole_x = center_x - p[0]
            from_pixel_to_pinhole_y = center_y - p[1]
            from_pixel_to_pinhole_z = -f
            
            # we should revert x and y here, but (see line 90)
            
            # Rotate ray to World Space using ONLY the BEV camera's absolute orientation 
            pitch = -np.pi/2
            yaw   = 0  
            roll  = 0  
            
            # 1. Pitch (X-axis)
            x1 = from_pixel_to_pinhole_x
            y1 = from_pixel_to_pinhole_y * np.cos(pitch) - from_pixel_to_pinhole_z * np.sin(pitch)
            z1 = from_pixel_to_pinhole_y * np.sin(pitch) + from_pixel_to_pinhole_z * np.cos(pitch)

            # 2. Yaw (Y-axis)
            x2 = x1 * np.cos(yaw) + z1 * np.sin(yaw)
            y2 = y1
            z2 = -x1 * np.sin(yaw) + z1 * np.cos(yaw)

            # 3. Roll (Z-axis)
            rot_x = x2 * np.cos(roll) - y2 * np.sin(roll)
            rot_y = x2 * np.sin(roll) + y2 * np.cos(roll)
            rot_z = z2
            
            # find the intersection of the ray with the floor plane (y=0)
            t = cam_bev[1] / -rot_y
            
            # If t is negative, the ray is shooting backwards up into the sky
            if t < 0:
                continue
            
            # got t, scale all of them to find the absolute world coordinates on the floor
            floor_x = cam_bev[0] + rot_x * t
            floor_y = 0.0
            floor_z = cam_bev[2] + rot_z * t
            
            # project them to the front camera
            # get the vector from the floor to the front camera pinhole
            from_floor_to_pinhole_x = cam_front[0] - floor_x
            from_floor_to_pinhole_y = cam_front[1] - floor_y
            from_floor_to_pinhole_z = cam_front[2] - floor_z

            # scale down to the focal length (similar triangles)
            # The signs naturally resolve because of the vector direction
            front_cam_x = from_floor_to_pinhole_x * f / from_floor_to_pinhole_z
            front_cam_y = from_floor_to_pinhole_y * f / from_floor_to_pinhole_z 
            
            # calculate the final pixel coordinates on the front image
            u_front = center_x + front_cam_x
            v_front = center_y + front_cam_y
            
            # we should use minus here, but since the rays have been through two inversions, we can just use plus here (see line 41)
            
            # record them
            new_pixels.append([int(u_front), int(v_front)])

        # Cast to np.int32 so cv2.fillPoly doesn't crash during drawing
        return np.array(new_pixels, dtype=np.int32)

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """
        new_image = cv2.fillPoly(self.image.copy(), [new_pixels], color)
        new_image = cv2.addWeighted(new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        points.append([x, y])
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        print(x, ' ', y)
        cv2.imshow('image', img)

points = []

if __name__ == "__main__":
    pitch_ang = -90

    front_rgb = "bev_data/front1.png"
    top_rgb = "bev_data/bev1.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)

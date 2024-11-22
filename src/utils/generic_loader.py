import os
import pickle
import numpy as np
from PIL import Image
from utils import *
import json

from utils import utilities
from utils.object_information import *

class GenericLoader:

    def __init__(self, obj_id: str, name: str, is_sim=True, mult_files = True, is_los=True, exp_num=1):
        """
        Initializes GenericLoader for loading robot and radar data

        Parameters:
            obj_id (str): id of the object (e.g. "001")
            name (str): name of the object (e.g. "phillips_screw_driver")
            is_sim (bool): True if gathering data from simulation environment
        """
        
        self.obj_id = obj_id
        self.name = name
        self.is_sim = is_sim
        self.mult_files = True
        self.is_los = is_los
        if exp_num == 'None' or exp_num is None: exp_num = 1
        self.exp_num = exp_num

    def get_path(self, radar_type=None, x_angle=0, y_angle=0, z_angle=0, is_processed=False, is_json=False):
        """
        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz","camera"). If left blank,
                only the path until the specified angle will be returned.
            angle (int): rotation angle in degrees of the loading object
            is_processed (bool): if True, will point to the processed folder
            is_json (bool): if True, return experiment json path

        Returns:
            A string that is the path to the directory containing the data for the specific radar type
        """
        dir_path = utilities.get_root_path()
        obj_path = self.obj_id + '_' + self.name
        sim_path = "simulation" if self.is_sim else "robot_collected"
        angle_path = str(x_angle) + "_" + str(y_angle) + "_" + str(z_angle)
        los_path = "" if self.is_sim or self.is_los is None else ("los" if self.is_los else "nlos")
        processed_path = "processed" if is_processed else "unprocessed"
        path_to_processed = f"{dir_path}/data/{obj_path}/{sim_path}/{angle_path}/{los_path}/{processed_path}"
        exp_path = f'exp{self.exp_num}' if not self.is_sim else ''
        if radar_type is None:
            raise ValueError("You should not use Radar_type = None")
            return path_to_processed 
        elif is_json:
            return f"{dir_path}/data/{obj_path}/{sim_path}/{angle_path}/{exp_path}"
        elif radar_type == "camera":
            camera_path = f"{dir_path}/data/{obj_path}/{sim_path}/{angle_path}/{exp_path}/{los_path}/{processed_path}/camera"
            return camera_path
        else:
            path_to_processed = f"{dir_path}/data/{obj_path}/{sim_path}/{angle_path}/{exp_path}/{los_path}/{processed_path}/radars"
            return f"{path_to_processed}/{radar_type}"

    def get_path_to_mesh(self, obj=False):
        """
        Parameters:
            obj (bool): if True, use obj file instead of stl file

        Returns:
            A string that is the path to the .obj or .stl file that contains the mesh file
        """ 
        assert self.is_sim , "Mesh files are only used for simulation. Use GenricLoader with is_sim=True"
        
        dir_path = utilities.get_root_path() 
        mesh_path = f"{dir_path}/data/{self.obj_id}_{self.name}"
        return f"{mesh_path}/textured.obj" if obj else f"{mesh_path}/nontextured.stl"

    def get_path_to_uniform_mesh(self, num_vert=None):
        assert self.is_sim , "Mesh files are only used for simulation. Use GenricLoader with is_sim=True"
        
        dir_path = utilities.get_root_path()
        pcd_path = f"{dir_path}/data/{self.obj_id}_{self.name}"
        ext = '' if num_vert is None else f'_{num_vert}'
        return f"{pcd_path}/uniform{ext}.stl"
        # return f"{pcd_path}/uniform.stl"
    
    def get_radar_paths(self, radar_type, x_angle=0, y_angle=0, z_angle=0):
        """
        Prints out and returns all paths to the radar files corresponding to the inputs

        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            angle (int): rotation angle in degrees of the loading object

        Returns:
            a list of strings of filepaths to all the radar data
        """
        path = self.get_path(radar_type, x_angle, y_angle, z_angle, is_processed=False) + "/radar_data"
        paths = []
        for filename in os.listdir(path):
            if filename[-4:] == ".bin":
                # print(f"{path}/{filename}")
                paths.append(f"{path}/{filename}")
        return paths

    def find_obj_angles(self):
        """
        Finds the object angles for the given experiment number. 
        This function will iterate through each angle folder to find the matching exp folder. 
        This assumes the matching exp data has been downloaded from AWS.

        Returns:
            a list of integers cooresponding to the X, Y, Z angles of the object for this experiment
        """
        if self.is_sim: return None
        obj_path = self.obj_id + '_' + self.name
        data_folder = f'{utilities.get_root_path()}/data/{obj_path}/robot_collected'
        for angle_folder in os.listdir(data_folder):
            for exp_folder in os.listdir(f'{data_folder}/{angle_folder}'):
                if exp_folder == f'exp{self.exp_num}': 
                    angles = [int(angle) for angle in angle_folder.split('_')] # Convert folder name to list of 3 angles
                    return angles 
        raise Exception(f"Couldn't find any folder matching exp{self.exp_num} for object {self.obj_id}_{self.name}")

    def save_image(self, radar_type, image, x_locs, y_locs, z_locs, x_angle=0, y_angle=0, z_angle=0, antenna_locs=None, ext= ""):
        """
        Saves processed image to the specified directory

        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            image (numpy array): processed image by ImageProcessor
            x_locs (numpy array): sampled x locations of image
            y_locs (numpy array): sampled y locations of image
            z_locs (numpy array): sampled z locations of image
            x/y/z angle (int): rotation angle in degrees of the object
            antenna_locs (numpy array): antenna locations used to processed image
            ext (str): extension of saved file name
        """
        data = {"image": image, "x_locs": x_locs, "y_locs": y_locs, "z_locs": z_locs, "antenna_locs": np.array(antenna_locs)}
        # path = self.get_path(radar_type, x_angle, y_angle, z_angle, is_processed=True) + "/test.pkl" #"/processed_image"+extra+".pkl"
        path = self.get_path(radar_type, x_angle, y_angle, z_angle, is_processed=True) + "/processed_image"+ext+".pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    def save_EMPTY_image(self, radar_type, image, x_locs, y_locs, z_locs, x_angle=0, y_angle=0, z_angle=0, antenna_locs=None):
        """
        Saves processed image of empty background to the specified directory

        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            image (numpy array): processed image by ImageProcessor
            x_locs (numpy array): sampled x locations
            y_locs (numpy array): sampled y locations
            z_locs (numpy array): sampled z locations
            angle (int): rotation angle in degrees of the loading object
        """
        data = {"image": image, "x_locs": x_locs, "y_locs": y_locs, "z_locs": z_locs, "antenna_locs": antenna_locs}
        path = self.get_path(radar_type, x_angle, y_angle, z_angle, is_processed=True) + "/test.pkl" #"/processed_EMPTY_image.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def save_rgbd(self, color, depth, x_angle=0, y_angle=0, z_angle=0):
        """
        Saves generated rgbd image to the specified directory

        Parameters:
            color (numpy array): generated color buffer
            depth (numpy array): generated depth buffer
            angle (int): rotation angle in degrees of the loading object
        """
        rgb_path = self.get_path(radar_type="camera", x_angle=x_angle, y_angle=y_angle, z_angle=z_angle, is_processed=False) + "/mesh_rgb.png"
        depth_path = self.get_path(radar_type="camera", x_angle=x_angle, y_angle=y_angle, z_angle=z_angle, is_processed=False) + "/mesh_depth.png"
        os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
        os.makedirs(os.path.dirname(depth_path), exist_ok=True)

        # Save rgb color buffer
        plt.figure(figsize=(8,8), frameon=False)
        plt.axis("off")
        plt.imshow(color)
        plt.savefig(rgb_path)

        # Save grayscale depth buffer
        plt.figure(figsize=(8,8), frameon=False)
        plt.axis("off")
        plt.imshow(depth, cmap=plt.cm.gray_r)
        plt.savefig(depth_path)
        plt.close("all")

    def save_raw_sim_files(self, radar_type, channels, radar_shifted_locs, x_angle=0, y_angle=0, z_angle=0, ext=""):
        """
        Saves raw simulation wave files to the specified directory

        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            channels (list(list)): wave channel information from simulation
            radar_shifted_locations (list(list)): radar shifted locations information from simulation
            angle (int): rotation angle in degrees of the loading object
        """
        data = {"channels": channels, "radar_shifted_locs": radar_shifted_locs}
        path = self.get_path(radar_type, x_angle, y_angle, z_angle, is_processed=False) + f"/raw_sim_files{ext}.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
    def load_raw_sim_files(self, radar_type, x_angle=0, y_angle=0, z_angle=0, ext=""):
        """
        loads raw simulation wave files 

        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            angle (int): rotation angle in degrees of the loading object
        """
        path = self.get_path(radar_type, x_angle, y_angle, z_angle, is_processed=False) + f"/raw_sim_files{ext}.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
        
    def load_image_file(self, radar_type, x_angle=0, y_angle=0, z_angle=0, background_subtraction=None, ext="", crop=False, crop_high=False):
        """
        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            angle (int): rotation angle in degrees of the loading object

        Returns: 
            Loaded processed image pickle file generated by ImageProcessor
        """
        if self.is_sim: background_subtraction = False
        if not self.is_sim:
            path = self.get_path(radar_type, x_angle, y_angle, z_angle, is_processed=True) + f"/processed_image{ext}.pkl"


        else:
            path = self.get_path(radar_type, x_angle, y_angle, z_angle, is_processed=True) + (f"/processed_image{ext}.pkl")

        with open(path, "rb") as f:
            data = pickle.load(f)
        image = data["image"]
        x_locs = data["x_locs"]
        y_locs = data["y_locs"]
        z_locs = data["z_locs"]
        antenna_locs = np.array(data["antenna_locs"])
        if crop:
            if radar_type=='77_ghz' and not self.is_sim:
                if not crop_high:
                    image = image[:,:,12:]
                    z_locs = z_locs[12:]
                    image = image[:,:,:-7]
                    z_locs = z_locs[:-7]
                else:
                    image = image[:,:,17:]
                    z_locs = z_locs[17:]
                    image = image[:,:,:-3]
                    z_locs = z_locs[:-3]
            else:
                image = image[:,:,:-3]
                z_locs = z_locs[:-3]
        if background_subtraction is not None and radar_type=='24_ghz':
            bg_loader = GenericLoader(background_subtraction, 'EMPTY', is_sim=False, is_los=True, exp_num=self.exp_num)
            bg_image, (bg_x_locs, bg_y_locs, bg_z_locs), _ = bg_loader.load_image_file('24_ghz', 0,0,0, ext=ext, crop=crop, crop_high=crop_high)
            if bg_image.shape[2] != image.shape[2]:
                diff = image.shape[2] -  bg_image.shape[2]
                image = image[:,:,diff:]
            extra =  np.array(bg_image.shape) - np.array(image.shape)
            if extra[0] != 0:
                print('f')
                image =np.abs(image) - np.abs( bg_image[:-extra[0]])
            else:
                print('j')
                image= np.abs(image) - np.abs(bg_image)
                
                

        return image, (x_locs, y_locs, z_locs), antenna_locs
    
    def load_camera_data(self):
        x, y, z = self.find_obj_angles()
        path = self.get_path(radar_type="camera", is_processed=False, x_angle=x, y_angle=y, z_angle=z)
        if not os.path.exists(path):
            print(f'Couldnt find object: {path}')
            return None
        filenames = sorted(os.listdir(path))
        for filename in filenames:
            if filename[-4:] != '.pkl' or filename[:10] != 'continuous':
                continue
            with open(f'{path}/{filename}', 'rb') as f:
                cam_data = pickle.load(f)
            break

        return cam_data

    def load_camera_masks(self, ext=''):
        x, y, z = self.find_obj_angles()
        path = self.get_path(radar_type="camera", is_processed=True, x_angle=x, y_angle=y, z_angle=z)
        path = f'{path}/camera_masks{ext}.pkl'
        if not os.path.exists(path):
            print(f'Couldnt find object: {path}')
            return None
        with open(f'{path}', 'rb') as f:
            masks = pickle.load(f)
        return masks
        
    def save_camera_masks(self, rgbd_mask, depth_masked, xyz_masked, cam_data, ext=''):
        path = self.get_path(radar_type="camera", is_processed=True)
        masks_path = f'{path}/camera_masks{ext}.pkl'
        os.makedirs(os.path.dirname(masks_path), exist_ok=True)
        data = {
            'rgbd_mask': rgbd_mask,
            'masked_depth': depth_masked,
            'pose': cam_data['pose'],
            'masked_xyz': xyz_masked,
        }
        with open(masks_path, "wb") as f:
            pickle.dump(data, f)
               
    def save_radar_masks(self, mask, radar_type='77_ghz', is_processed=True, ext='', x_angle=0, y_angle=0, z_angle=0):
        path = self.get_path(radar_type=radar_type, is_processed=is_processed, x_angle=x_angle, y_angle=y_angle, z_angle=z_angle)
        masks_path = f'{path}/radar_masks{ext}.pkl'
        os.makedirs(os.path.dirname(masks_path), exist_ok=True)
        data = {
            'radar_mask': mask
        }
        with open(masks_path, "wb") as f:
            pickle.dump(data, f)

    def load_radar_masks(self, radar_type='77_ghz', is_processed=True, ext='', x_angle=0, y_angle=0, z_angle=0):
        path = self.get_path(radar_type=radar_type, is_processed=is_processed, x_angle=x_angle, y_angle=y_angle, z_angle=z_angle)
        masks_path = f'{path}/radar_masks{ext}.pkl'
        if not os.path.exists(masks_path):
            print(f'Couldnt find object: {masks_path}')
            return None
        with open(f'{masks_path}', 'rb') as f:
            masks = pickle.load(f)
        return masks

    def load_radar_files(self, radar_type, x_angle=0, y_angle=0, z_angle=0, aperture_type='normal'):
        """
        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            angle (int): rotation angle in degrees of the loading object

        Returns: 
            Loaded complex-valued radar (adc) files for the specified radar type
        """
        path = self.get_path(radar_type, x_angle, y_angle, z_angle, is_processed=False) + "/radar_data"
        all_data = {}
        filenames = sorted(os.listdir(path))
        params_dict = utilities.get_radar_parameters(radar_type=radar_type, is_sim=False, aperture_type=aperture_type)
        NUM_FRAMES = params_dict['num_frames']
        SAMPLES_PER_CHIRP = params_dict['num_samples']
        NUM_CHIRP = params_dict['num_chirps']
        for i, filename in enumerate(filenames):
            if radar_type == "24_ghz":

                # 24 GHz needs formatting before loading
                if filename[-4:] != '.txt': continue
                timestamp = int(filename[9:19])
                try:
                    adcData = np.loadtxt(f'{path}/{filename}', dtype=None, delimiter=',')
                except:
                    continue

                adcDataFormatted = np.zeros(((SAMPLES_PER_CHIRP * NUM_CHIRP * NUM_FRAMES), 2), dtype=np.complex64)
                if adcData.shape[0] == NUM_CHIRP * NUM_FRAMES*4:
                    for i in range(NUM_CHIRP * NUM_FRAMES):
                        adcDataFormatted[SAMPLES_PER_CHIRP * i: SAMPLES_PER_CHIRP * (i + 1), 0] = adcData[i * 4] + 1j * adcData[i * 4 + 1]
                        adcDataFormatted[SAMPLES_PER_CHIRP * i: SAMPLES_PER_CHIRP * (i + 1), 1] = adcData[i * 4 + 2] + 1j * adcData[i * 4 + 3]
                all_data[timestamp] = adcDataFormatted

            elif radar_type == "77_ghz":

                if filename[-4:] != '.bin': continue
                try:
                    timestamp = int(filename[8:18])
                except:
                    timestamp = int(filename[9:21])

                fid = open(f'{path}/{filename}', 'rb')
                adcData = np.fromfile(fid, dtype='<i2')
                numLanes = 4
                adcData = np.reshape(adcData, (int(adcData.shape[0] / (numLanes * 2)), numLanes * 2))
                adcData = adcData[:, [0, 1, 2, 3]] + 1j * adcData[:, [4, 5, 6, 7]]
                all_data[timestamp] = adcData

        return all_data

    def load_robot_loc_files(self, radar_type, x_angle=0, y_angle=0, z_angle=0):
        """
        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            angle (int): rotation angle in degrees of the loading object

        Returns:
            Loaded continuous robot location (antenna) numpy (npy) files
        """
        path = self.get_path(radar_type, x_angle, y_angle, z_angle, is_processed=False)
        if self.mult_files:
            robot_loc_files = []
            for filename in os.listdir(path):
                if filename[:7] == 'antenna':
                    robot_loc_files.append(filename)
            combined = {}
            robot_loc_files = sorted(robot_loc_files)
            for filename in robot_loc_files:
                cur_file = dict(np.load(f'{path}/{filename}'))
                if cur_file['times_77'].shape[0] != cur_file['tx_77_locs'].shape[0]:    
                    for key in cur_file:
                        min_val = min(cur_file['times_77'].shape[0], cur_file['tx_77_locs'].shape[0])
                        cur_file['times_77'] = cur_file['times_77'][:min_val-20]
                        cur_file['tx_77_locs'] = cur_file['tx_77_locs'][:min_val-20]
                if cur_file['times_24'].shape[0] != cur_file['tx_24_locs'].shape[0]:    
                    for key in cur_file:
                        min_val = min(cur_file['times_24'].shape[0], cur_file['tx_24_locs'].shape[0])
                        cur_file['times_24'] = cur_file['times_24'][:min_val-20]
                        cur_file['tx_24_locs'] = cur_file['tx_24_locs'][:min_val-20]
                        
                for key in cur_file:
                    try:
                        combined[key] = np.concatenate((combined[key], cur_file[key]))
                    except KeyError:
                        combined[key] = cur_file[key]
            return combined
        else:
            for filename in os.listdir(path):
                if filename[:7] == 'antenna': break
            return dict(np.load(f'{path}/{filename}'))

    def load_robot_file(self, radar_type, x_angle=0, y_angle=0, z_angle=0):
        """
        Loads robot pickle file labeled "simple" or "continuous." Simple robot data are 
        collected as the robot repeatedly moves and stops to collect data. Continuous robot
        data are collected as the robot continuously moves on each row.

        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            angle (int): rotation angle in degrees of the loading object

        Returns:
            Dictionary of robot data
        """
        path = self.get_path(radar_type, x_angle, y_angle, z_angle, is_processed=False)
        if self.mult_files:
            robot_files = []
            for filename in os.listdir(path):
                if filename[:10] == 'continuous':
                    robot_files.append(filename)
            robot_files = sorted(robot_files)
            robot_data = {}
            for filename in robot_files:
                with open(f'{path}/{filename}', 'rb') as f:
                    cur_file = pickle.load(f)
                    for key in cur_file:
                        if key == 'pattern':
                            continue
                        try:
                            robot_data[key] = robot_data[key] + cur_file[key]
                        except KeyError:
                            robot_data[key] = cur_file[key]
                    

            robot_data['pattern'] = 0.5
        else:
            for filename in os.listdir(path):
                if filename[:10] == 'continuous': break
            with open(f'{path}/{filename}', 'rb') as f:
                robot_data = pickle.load(f)

        return robot_data
    
    def load_exp_json(self, radar_type, x_angle, y_angle, z_angle):
        """ 
        Load the experiment json file.

        Parameters: 
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            x/y/z angle (int): rotation angle in degrees of the loading object
        Returns: 
            json data as a dictionary
        """
        path_to_json = self.get_path(radar_type, x_angle, y_angle, z_angle, is_json=True)

        # Open and read the JSON file
        with open(f'{path_to_json}/experiment_info.json', 'r') as file:
            data = json.load(file)
        return data

    def load_all_data(self, radar_type, x_angle=0, y_angle=0, z_angle=0):
        """
        Loads all data: robot location, radar data, and continuous robot location data
    
        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            angle (int): rotation angle in degrees of the loading object

        Returns:
            Dictionaries of robot data, radar data, and continuous robot location data
        """
        exp_data = self.load_exp_json(radar_type, x_angle, y_angle, z_angle)
        aperture_type = exp_data['aperture size']
        robot_data = self.load_robot_file(radar_type, x_angle, y_angle, z_angle)
        radar_data = self.load_radar_files(radar_type, x_angle, y_angle, z_angle, aperture_type=aperture_type)
        robot_loc_data = self.load_robot_loc_files(radar_type, x_angle, y_angle, z_angle)
        return robot_data, radar_data, robot_loc_data, exp_data

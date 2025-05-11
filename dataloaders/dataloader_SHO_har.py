import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA
# ========================================       SHO_HAR_DATA             =============================
class SHO_HAR_DATA(BASE_DATA):
    """
    Fusion of Smartphone Motion Sensors for Physical Activity Recognition
    https://www.utwente.nl/en/eemcs/ps/research/dataset/

    Brief Description of the Dataset:
    ---------------------------------
    In the data collection experiments, we collected data for seven physical activities. 
    These are walking, sitting, standing, jogging, biking, walking upstairs and walking downstairs, 
    which are mainly used in the related studies and they are the basic motion activities in daily life.
    There were ten participants involved in our data collection experiment who performed each of these activities for 3-4 minutes. 
    All ten participants were male, between the ages of 25 and 30. 
    The experiments were carried out indoors in one of the university buildings, except biking. 
    For walking, and jogging, the department corridor was used. 
    For walking upstairs and downstairs, a 5-floor building with stairs was used. 
    Each of these participants was equipped with five smartphones on five body positions: 
        One in their right jean pocket. 
        One in their left jean pocket.
        One on belt position towards the right leg using a belt clipper.
        One on the right upper arm. 
        One on the right wrist. 
    The data was recorded for all five positions at the same time for each activity and it was collected at a rate of 50 samples per second. 
    This sampling rate (50 samples per second) is enough to recognize human physical activities, as we show in our previous study . 
    Moreover, in the state of the art, frequencies lower than 50 samples per second have been shown to be sufficient for activity recognition.
    """

    def __init__(self, args):

        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """

        # Because there are empty cols 
        self.used_cols = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, # Left_pocket
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, # Right_pocket
                         29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, # Wrist
                         43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, # Upper_arm
                         57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, # Belt
                         69]

        col_list    =  ["acc_x","acc_y","acc_z","lacc_x","lacc_y","lacc_z","Gyro_x","Gyro_y","Gyro_z","mag_x","mag_y","mag_z"]
        pos_list = ["Left_pocket", "Right_pocket", "Wrist", "Upper_arm", "Belt"]
        self.col_names    =  [item for sublist in [[col+"_"+pos for col in col_list] for pos in pos_list] for item in sublist]+["activity_id"]

        # The original file used the str as the label, they need first encoded
        self.label_map = [(0, 'walking'), 
                          (1, 'standing'),
                          (2, 'jogging'),
                          (3, 'sitting'), 
                          (4, 'biking'),
                          (5, 'upstairs'),
                          (6, "downstairs")]

        self.drop_activities = []

        # TODO This should be referenced by other paper
        self.train_keys   = [1, 2, 3, 4, 5, 6, 7, 8]
        self.vali_keys    = []
        self.test_keys    = [9, 10]


        self.LOCV_keys = [[1,2],[3,4],[5,6],[7,8], [9,10]]
        self.all_keys = [1,2,3,4,5,6,7,8,9,10]
        self.sub_ids_of_each_sub = {}


        self.exp_mode     = args.exp_mode
        self.split_tag = "sub"

        self.file_encoding = {"Participant_1.csv":1, "Participant_2.csv":2, "Participant_3.csv":3,
                              "Participant_4.csv":4, "Participant_5.csv":5, "Participant_6.csv":6,
                              "Participant_7.csv":7, "Participant_8.csv":8, "Participant_9.csv":9,
                              "Participant_10.csv":10} 


        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(SHO_HAR_DATA, self).__init__(args)

    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")

        file_list = os.listdir(root_path)
        file_list = [file for file in file_list if "Participant" in file] # in total , it should be 10

        assert len(file_list) == 10

        df_dict = {}

        for file in file_list:
            sub_data = pd.read_csv(os.path.join(root_path, file), skiprows=[0,1], header=None)
            sub_data =sub_data.iloc[:,self.used_cols]

            sub_data.columns = self.col_names

            # if missing values, imputation TODO
            sub_data = sub_data.interpolate(method='linear', limit_direction='both')

            sub = int(self.file_encoding[file])
            sub_data['sub_id'] = sub
            sub_data["sub"] = sub

            if sub not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub] = []
            self.sub_ids_of_each_sub[sub].append(sub)

            df_dict[sub] = sub_data

        df_all = pd.concat(df_dict)
        df_all = df_all.set_index('sub_id')

        # this is for first encode the str label to nummeric number
        label_mapping = {item[1]:item[0] for item in self.label_map}

        # there are typos in the data!!!
        def change_the_typo(label):
            if label == "upsatirs":
                return "upstairs"
            else:
                return label

        df_all["activity_id"] = df_all["activity_id"].apply(change_the_typo)
        df_all["activity_id"] = df_all["activity_id"].map(label_mapping)
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        df_all = df_all[self.col_names[:-1]+["sub"]+["activity_id"]]

        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 

        return data_x, data_y
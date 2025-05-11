import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA

# ========================================       WISDM_HAR_DATA             =============================
class RECGYM_HAR_DATA(BASE_DATA):

    """

    https://zhaxidele.github.io/RecGym/

    """

    def __init__(self, args):


        # There is only one file which includs the collected data from 33 users
        # delete the second column it is the timestamp
        self.used_cols    = [0,3,4,5,6,7,8,9,10]
        self.col_names    =  ["Subject", "Position", "Session", "A_x", "A_y", "A_z", "G_x", "G_y", "G_z", "C_1", "Workout"]

        # pos_filter ------- >  filter according to position
        # sensor_filter ----->  filter according to the sensor type
        self.pos_filter         = None
        self.sensor_filter      = None

        # selected_cols will be updated according to user settings. User have to set -- args.pos_select, args.sensor_select---
        self.selected_cols      = None
        # Filtering channels according to the Position
        self.selected_cols      = self.Sensor_filter_acoording_to_pos_and_type(args.pos_select, self.pos_filter, self.col_names, "position")
        # Filtering channels according to the Sensor Type
        if self.selected_cols is None:
            self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.col_names, "Sensor Type")
        else:
            self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.selected_cols, "Sensor Type")




        self.label_map = [(0, 'Adductor'),
                          (1, 'ArmCurl'),
                          (2, 'BenchPress'), 
                          (3, 'LegCurl'),
                          (4, 'LegPress'),
                          (5, 'Null'),
                          (6, 'Riding'),
                          (7, 'RopeSkipping'),
                          (8, 'Running'),
                          (9, 'Squat'),
                          (10, 'StairClimber'),
                          (11, 'Walking')]


        self.drop_activities = []
        # TODO This should be referenced by other paper
        # TODO , here the keys for each set will be updated in the readtheload function

        #self.train_keys   = [1,2,3,4,5,  7,8,9,10,11,  13,14,15,16,17,  19,20,21,22,23,  25,26,27,28,29,  31,32,34,35,36]
        #self.vali_keys    = []
        #self.test_keys    = [6,12,18,24,30,33]
        self.train_keys   = [1,2,3,4,5,6,7,8,9]
        self.vali_keys    = []
        self.test_keys    = [10]

        self.exp_mode     = args.exp_mode

        self.split_tag = "sub"

        self.LOCV_keys = [[1],[2],[3],[4],[5],[6],[7],[8],[9]]
        self.all_keys = [1,2,3,4,5,6,7,8,9,10]
        self.sub_ids_of_each_sub = {}

        self.file_encoding = {} # no use

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(RECGYM_HAR_DATA, self).__init__(args)


    def load_all_the_data(self, root_path):
        print(" ----------------------- load all the data -------------------")
        #df_all = pd.read_csv(os.path.join(root_path,"RecGym.csv"),header=None,names=self.col_names)
        df_all = pd.read_csv(os.path.join(root_path,"RecGym.csv"))
        print(df_all)
        df_all = df_all[df_all["Position"]=="wrist"]
        df_all =df_all.iloc[:,self.used_cols]
        df_all.dropna(inplace=True)
        print(df_all)

        df_all['act_block'] = ( (df_all['Subject'].shift(1) != df_all['Subject'])).astype(int).cumsum()
        print(df_all)
        sub_id_list = []
        for index in df_all.act_block.unique():
            temp_df = df_all[df_all["act_block"]==index]
            sub = temp_df["Subject"].unique()[0]
            sub_id = "{}_{}".format(sub,index)
            sub_id_list.extend([sub_id]*temp_df.shape[0])

            if sub not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub] = []
            self.sub_ids_of_each_sub[sub].append(sub_id)    

        df_all["sub_id"] =     sub_id_list
        del df_all["act_block"]
        df_all=df_all.rename(columns = {'Subject':'sub'})

        label_mapping = {'Adductor':0, 
                         'ArmCurl':1,
                          'BenchPress':2,
                          'LegCurl':3, 
                          'LegPress':4,
                          'Null':5,
                          'Riding':6,
                          'RopeSkipping':7,
                          'Running':8,
                          'Squat':9,
                          'StairClimber':10,
                          'Walking':11}

        df_all["activity_id"] = df_all["Workout"].map(label_mapping)
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        df_all = df_all.set_index('sub_id')

        if self.selected_cols:
            df_all = df_all[self.selected_cols+["sub"]+["activity_id"]]
        else:
            df_all = df_all[["A_x","A_y","A_z", "G_x","G_y","G_z", "C_1"]+["sub"]+["activity_id"]]


        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]
        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 
        print(data_x)
        print(data_y)
        return data_x, data_y
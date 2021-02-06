"""
Owner: isst
Please implement your own ModelImp
"""
import os
import utils

import sys


class CmdLineArgs:
    def __init__(self):
        base_dir = utils.get_model_path()
        model_dir = os.path.join(os.path.dirname(os.path.realpath(base_dir)), "model/model_dir")

        self.config_file = os.path.join(model_dir, "capt_qna_1n.json")
        self.model_checkpoint = os.path.join(model_dir, "resume_model_15_1.0.tar")
        self.gv = os.path.join(model_dir, "300kG_ConsecutiveData_1occ_train_10k.gv")

        self.use_cpu = False
        self.local_rank = -1
        self.max_sentences = 200
        self.max_sentence_length = 50
        self.max_chars = 350
        self.max_out_sentences = 3


class ModelImp:
    # self.model_path is "Model Path" specified when deploying the model (i.e. <path1>/run.sh)
    #
    # self.model_dir is directory where model is located under "Model Path" (i.e. <path1>/model/)
    #
    # self.data_path is the "Model Data Path" specified when deploying the model (i.e. <path2>/data.txt)
    #   If it was not specified during deployment, it will be none
    #
    # self.data_dir is the directory containing the file specified in data_path (i.e. <path2>)
    #
    # To access files in /model, use self.model_dir
    # To access files in the data folder, use self.data_dir
    def  __init__(self):
        self.model_path = utils.get_model_path()
        self.model_dir = os.path.join(os.path.dirname(os.path.realpath(self.model_path)), "model")

        self.data_path = utils.get_data_path()
        self.data_dir = None
        if self.data_path is not None:
            self.data_dir = os.path.dirname(os.path.realpath(self.data_path))

        print("Model Path: {model_path}".format(model_path=self.model_path))
        print("Model Dir: {model_dir}".format(model_dir=self.model_dir))
        print("Data Path: {data_path}".format(data_path=self.data_path))
        print("Data Dir: {data_dir}".format(data_dir=self.data_dir))

        sys.path.insert(0, self.model_dir)
        import model_instance

        print('loading model...')

        args = CmdLineArgs()
        self.model_instance = model_instance.ModelExecutor(args)
        print('model loaded.')

    # string version of Eval
    # data is a string
    def Eval(self, data):
        return self.model_instance.Eval(data)


    # binary version of Eval
    # data is python class "bytes"
    def EvalBinary(self, data):
        # Implement your binary evaluation here
        return data

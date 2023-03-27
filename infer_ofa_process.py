# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
from ikomia import core, dataprocess
from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator
import os
import subprocess
import sys


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferOfaParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        # Example : self.windowSize = 25
        self.size = "Tiny"
        self.update = True

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(param_map["windowSize"])
        self.size = param_map["size"]
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        # Example : paramMap["windowSize"] = str(self.windowSize)
        param_map["size"] = self.size
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferOfa(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        # Example :  self.add_input(dataprocess.CImageIO())
        #           self.add_output(dataprocess.CImageIO())
        self.tokenizer = None
        self.model = None
        self.generator = None

        # Create parameters class
        if param is None:
            self.set_param_object(InferOfaParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def download_if_necessary(self, model_size):
        model_name = "OFA-"+model_size
        work_dir = os.path.dirname(__file__)
        folder = os.path.join(work_dir, model_name)
        if os.path.isdir(folder):
            return
        else:
            platform = sys.platform
            if platform == "linux":
                subprocess.run("git lfs install", shell=True, check=True)
                print("Downloading {} weights...".format(model_name))
                subprocess.run('cd {}; git clone https://huggingface.co/OFA-Sys/OFA-{}'.format(work_dir, model_size), shell=True, check=True)
            if platform == "windows":
                subprocess.run("git lfs install", shell=True, check=True)
                print("Downloading {} weights...".format(model_name))
                subprocess.run('dir {}^ git clone https://huggingface.co/OFA-Sys/OFA-{}'.format(work_dir, model_size),
                               shell=True, check=True)


    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        param = self.get_param_object()

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 256
        patch_resize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.Normalize(mean=mean, std=std)
        ])

        if self.model is None or param.update:
            self.download_if_necessary(param.size)
            model_name = "OFA-" + param.size
            work_dir = os.path.dirname(__file__)
            ckpt_dir = os.path.join(work_dir, model_name)

            self.tokenizer = OFATokenizer.from_pretrained(ckpt_dir)

            # using the generator of fairseq version
            self.model = OFAModel.from_pretrained(ckpt_dir, use_cache=True)
            self.generator = sequence_generator.SequenceGenerator(
                tokenizer=self.tokenizer,
                beam_size=5,
                max_len_b=16,
                min_len=0,
                no_repeat_ngram_size=3,
            )
            param.update = False

        txt = " what does the image describe?"
        inputs = self.tokenizer([txt], return_tensors="pt").input_ids
        img = self.get_input(0).get_image()
        patch_img = patch_resize_transform(img).unsqueeze(0)

        gen = self.model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3)

        print(self.tokenizer.batch_decode(gen, skip_special_tokens=True))

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferOfaFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_ofa"
        self.info.short_description = "your short description"
        self.info.description = "your description"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        self.info.version = "1.0.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "algorithm author"
        self.info.article = "title of associated research article"
        self.info.journal = "publication journal"
        self.info.year = 2021
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentation_link = ""
        # Code source repository
        self.info.repository = ""
        # Keywords used for search
        self.info.keywords = "your,keywords,here"

    def create(self, param=None):
        # Create process object
        return InferOfa(self.info.name, param)

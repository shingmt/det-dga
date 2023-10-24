import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# Copyright 2022 by Minh Nguyen. All rights reserved.
#     @author Minh Tu Nguyen
#     @email  nmtu.mia@gmail.com
# 
# Licensed under the Apache License, Version 2.0 (the "License"): you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

from worker.base.silentworker_base import SilentWorkerBase
from utils.utils import log
import pickle
from dga.combination_features import features_combination
import tldextract
from dga.combination_features import DICTIONARY

import warnings
warnings.simplefilter("ignore")


class SilentWorker(SilentWorkerBase):
    """
    This is the baseline for the SilentWorker.

    SilentWorker should be the one carrying the main work.
    Sometimes a module might take a great deal of time to generate outputs. 

    Whenever your model finishes an operation, make sure to call `__onFinish__` function to populate your results to db as well as to the next module in the chain.
        Command:
            self.__onFinish__(output[, note])

        whereas:
            - output (mandatory): A dict that maps a `orig_input_hash` => output
                Output could be of any type. Just make sure to note that in your module's configuration and description for others to interpret your results.
                For example,
                    - a detector module can return a boolean (to represent detection result).
                    - a non-detector module (eg. processor or sandbox) can return the path storing processed result.
                eg.
                    {
                        'hash_of_input_1': true,
                        'hash_of_input_2': false,
                        'hash_of_input_3': false,
                        ...
                    }
            - note (optional).
                It could be a dict or a string.
                If it's a dict, it must be a map of a filepath to a note (string). The system will find and save the note accordingly with the file. 
                If it's not a map, use a string. The system will save this note for all the files analyzed in this batch
    """

    def __init__(self, config, options=None):
        """ Dont change/remove this super().__init__ line.
            This line passes config to initialize services. 
        """
        super().__init__(config, options)

        #! Add your parts of initializing the model or anything you want here. 
        #! You might want to load everything at this init phase, not reconstructing everything at each request (which shall then be defined in run())
        print('Nooby doo')

    
    def onChangeConfig(self, config_data):
        """
        Callback function when module's config is changed.
        (usually at each request to analyze, when config_data is sent along as a parameter)
        ---
        This is the module's config which is passed at each analysis request. (usually the config to initialize the model)
        """

        log(f'[ ][SilentWorker][onChangeConfig] config_data is passed: {config_data}')

        #! Want to do something with the model when the config is changed ? Maybe reconfig the model's params, batch size etc. ?
        #? eg. change global module's config
        #self._config = config_data
        
        return


    def classifier(self, ohashes, inputs):
        #print(exam)
        # xu ly domain thanh vector

        max_features = 38
        maxlen = 57
        
        # print('exam', exam)
        
        X = [tldextract.extract(i).domain for i in inputs]
        
        with open(DICTIONARY, "rb") as f:
            dic = pickle.load(f)
        valid_chars = {x: idx + 1 for idx, x in enumerate(dic)}
        # print(valid_chars)

        '''Reloading the model
        load the model from Saved file'''
        Model_file = 'dga/model.pkl'
        loaded_model = pickle.load(open(Model_file, 'rb'))

        X = features_combination(X)

        result = {}
        K = loaded_model.predict(X)
        for i in range(len(K)):
            item = bool(int(K[i]))
            # print(i, ohashes[i], item)
            result[ohashes[i]] = item
        
        
        print('[+][classifier] res', result)

        #! Call __onFinishInfer__ when the analysis is done. This can be called from anywhere in your code. In case you need synchronous processing
        self.__onFinishInfer__(result)

        return result


    def infer(self, config_data):
        """
        #? This function is to be overwritten.
        Main `inference` function. 

            #? (used for all modules, `detector`, `(pre)processor`, `sandbox`)
            Whatever you need to do in silence, put it here.
            We provide inference in batch, for heavy models.

        ----
        Use these vars to access input data:
            - self._map_ohash_inputs: dict of inputs to feed this module (eg. filepath to the executable files already stored on the system / url).
                map `orig_input_hash` => `prev module's output for orig_path correspond with orig_input_hash`
            - self._map_ohash_oinputs: dict of original inputs that is fed to this module flow.
                map `orig_input_hash` => one `orig_path`.
                (multiple orig_path might have the same hash, but just map to one path)
        
        Params:
            - config_data: modules configuration stored in the db.
        """

        #! Do something
        log('[ ][SilentWorker][infer] I\'m pretty')

        try:
            self.classifier(list(self._map_ohash_inputs.keys()), list(self._map_ohash_inputs.values()))
        except Exception as e:
            log(f'[!][SilentWorker][infer] Failed with exception: {e}')

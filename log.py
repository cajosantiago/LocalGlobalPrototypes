import os
import argparse
import numpy as np
import torch
#from util.args import save_args

class Log:

    """
    Object for managing the log directory
    """

    def __init__(self, log_dir: str):  # Store log in log_dir

        self._log_dir = log_dir
        self._logs = dict()

        # Ensure the directories exist
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        open(self.log_dir + '/log.txt', 'w').close() #make log file empty if it already exists

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def checkpoint_dir(self):
        return self._log_dir + '/checkpoints'

    @property
    def metadata_dir(self):
        return self._log_dir + '/metadata'

    def log_message(self, msg: str):
        """
        Write a message to the log file
        :param msg: the message string to be written to the log file
        """
        with open(self.log_dir + '/log.txt', 'a') as f:
            f.write(msg+"\n")

    def create_log(self, log_name: str, key_name: str, *value_names):
        """
        Create a csv for logging information
        :param log_name: The name of the log. The log filename will be <log_name>.csv.
        :param key_name: The name of the attribute that is used as key (e.g. epoch number)
        :param value_names: The names of the attributes that are logged
        """
        if log_name in self._logs.keys():
            raise Exception('Log already exists!')
        # Add to existing logs
        self._logs[log_name] = (key_name, value_names)
        # Create log file. Create columns
        with open(self.log_dir + f'/{log_name}.csv', 'w') as f:
            f.write(','.join((key_name,) + value_names) + '\n')

    def log_values(self, log_name, key, *values):
        """
        Log values in an existent log file
        :param log_name: The name of the log file
        :param key: The key attribute for logging these values
        :param values: value attributes that will be stored in the log
        """
        if log_name not in self._logs.keys():
            raise Exception('Log not existent!')
        if len(values) != len(self._logs[log_name][1]):
            raise Exception('Not all required values are logged!')
        # Write a new line with the given values
        with open(self.log_dir + f'/{log_name}.csv', 'a') as f:
            f.write(','.join(str(v) for v in (key,) + values) + '\n')

#    def log_args(self, args: argparse.Namespace):
#        save_args(args, self._log_dir)

    def save_stuff(self, epoch, train_info, eval_info, best_test_acc, best_test_auc, model):
        if eval_info['test_bal_accuracy'] > best_test_acc:
            np.savetxt(os.path.join(self.log_dir,'log_best_test_cm.csv'), eval_info['confusion_matrix'], delimiter=",", fmt='%d')
            np.savetxt(os.path.join(self.log_dir,'log_best_test_auc.csv'), np.array([eval_info['test_auc']]), delimiter=",", fmt='%f')
            best_test_acc = eval_info['test_bal_accuracy']
            directory_path= os.path.join(self.log_dir, 'checkpoints', 'best_test_model')
            if not os.path.isdir(directory_path):
                os.mkdir(directory_path)
            torch.save(model.state_dict(), os.path.join(directory_path, 'model_state.pt'))
        if eval_info['test_auc'] > best_test_auc:
            np.savetxt(os.path.join(self.log_dir,'log_bestauc_test_cm.csv'), eval_info['confusion_matrix'], delimiter=",", fmt='%d')
            np.savetxt(os.path.join(self.log_dir,'log_bestauc_test_auc.csv'), np.array([eval_info['test_auc']]), delimiter=",", fmt='%f')
            best_test_auc = eval_info['test_auc']
            directory_path= os.path.join(self.log_dir, 'checkpoints', 'best_testauc_model')
            if not os.path.isdir(directory_path):
                os.mkdir(directory_path)
            torch.save(model.state_dict(), os.path.join(directory_path, 'model_state.pt'))
        self.log_values('log_epoch_overview', epoch, eval_info['test_bal_accuracy'], eval_info['test_accuracy'],
                       eval_info['test_accuracy_global'], eval_info['test_accuracy_local'], train_info['train_accuracy'],
                       train_info['train_accuracy_global'], train_info['train_accuracy_local'], train_info['loss'])

        if model.local_prototypes:
            path_local_count = os.path.join(self.log_dir, 'count', 'local')
            if not os.path.isdir(path_local_count):
                os.makedirs(path_local_count)
            np.savetxt(os.path.join(path_local_count, 'count_%03d.csv' % epoch), train_info['local_counter'].numpy(),
                       delimiter=",", fmt='%d')
        if model.global_prototypes:
            path_global_count = os.path.join(self.log_dir, 'count', 'global')
            if not os.path.isdir(path_global_count):
                os.makedirs(path_global_count)
            np.savetxt(os.path.join(path_global_count, 'count_%03d.csv' % epoch), train_info['global_counter'].numpy(), delimiter=",",
                       fmt='%d')

        return best_test_acc, best_test_auc

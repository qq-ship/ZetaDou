import numpy as np
from op_agent.sl.sl_data_formate import _get_obs_landlord,_get_obs_landlord_down,_get_obs_landlord_up
import torch

def _load_model(position, model_path):
    from op_agent.sl.models import model_dict
    model = model_dict[position]()
    model_state_dict = model.state_dict()
    if torch.cuda.is_available():
        pretrained = torch.load(model_path, map_location='cuda:0')
    else:
        pretrained = torch.load(model_path, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

class Slagent(object):

    def __init__(self,position,model_path):

        self.use_raw = False
        self.position = position
        self.model = _load_model(position, model_path)


    def step(self,state):



        if self.position == 'landlord':
            x_batch,z_batch = _get_obs_landlord(state)

            z_batch = torch.from_numpy(np.array(z_batch)).float().cuda()
            x_batch = torch.from_numpy(np.array(x_batch)).float().cuda()


            y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
            y_pred = y_pred.detach().cpu().numpy()

            best_action_index = np.argmax(y_pred, axis=0)[0]

            legal = list(state['legal_actions'].keys())
            best_action = legal[best_action_index]

            return best_action

        elif self.position == 'landlord_down':
            x_batch,z_batch = _get_obs_landlord_down(state)

            z_batch = torch.from_numpy(np.array(z_batch)).float().cuda()
            x_batch = torch.from_numpy(np.array(x_batch)).float().cuda()

            y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
            y_pred = y_pred.detach().cpu().numpy()

            best_action_index = np.argmax(y_pred, axis=0)[0]

            legal = list(state['legal_actions'].keys())
            best_action = legal[best_action_index]

            return best_action

        elif self.position == 'landlord_up':
            x_batch, z_batch = _get_obs_landlord_down(state)

            z_batch = torch.from_numpy(np.array(z_batch)).float().cuda()
            x_batch = torch.from_numpy(np.array(x_batch)).float().cuda()

            y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
            y_pred = y_pred.detach().cpu().numpy()

            best_action_index = np.argmax(y_pred, axis=0)[0]

            legal = list(state['legal_actions'].keys())
            best_action = legal[best_action_index]

            return best_action

    def eval_step(self, state):

        pa = 0
        return self.step(state),pa

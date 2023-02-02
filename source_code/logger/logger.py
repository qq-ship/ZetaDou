import matplotlib.pyplot as plt
import os

class Logger(object):
    ''' Logger saves the running results and helps make plots from the results
    '''

    def __init__(self, xlabel = '', ylabel = '',zlabel='',mlabel='',lslabel='',ps1label='',ps2label='',l_a_loss='',l_c_loss='',l_s_loss='',p1_a_loss='',p1_c_loss='',p1_s_loss='',p2_a_loss='',p2_c_loss='',p2_s_loss='',k_l_loss='',k_d_loss='',k_u_loss='',acc_0='',acc_1='',acc_2='',acc_t='',legend = '', log_path = None, csv_path = None):
        ''' Initialize the labels, legend and paths of the plot and log file.

        Args:
            xlabel (string): label of x axis of the plot
            ylabel (string): label of y axis of the plot
            legend (string): name of the curve
            log_path (string): where to store the log file
            csv_path (string): where to store the csv file

        Note:
            1. log_path must be provided to use the log() method. If the log file already exists, it will be deleted when Logger is initialized.
            2. If csv_path is provided, then one record will be write to the file everytime add_point() method is called.
        '''
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.mlabel = mlabel
        self.lslabel = lslabel
        self.ps1label = ps1label
        self.ps2label = ps2label
        self.l_a_loss = l_a_loss
        self.l_c_loss = l_c_loss
        self.l_s_loss = l_s_loss
        self.p1_a_loss = p1_a_loss
        self.p1_c_loss = p1_c_loss
        self.p1_s_loss = p1_s_loss
        self.p2_a_loss = p2_a_loss
        self.p2_c_loss = p2_c_loss
        self.p2_s_loss = p2_s_loss
        self.k_l_loss = k_l_loss
        self.k_d_loss = k_d_loss
        self.k_u_loss = k_u_loss


        self.legend = legend
        self.xs = []
        self.ys = []
        self.log_path = log_path
        self.csv_path = csv_path
        self.log_file = None
        self.csv_file = None
        if log_path is not None:
            log_dir = os.path.dirname(log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.log_file = open(log_path, 'w')
        if csv_path is not None:
            csv_dir = os.path.dirname(csv_path)
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            self.csv_file = open(csv_path, 'w')
            self.csv_file.write(xlabel+','+ylabel+','+zlabel+','+mlabel+','+lslabel+','+ps1label+','+ps2label+','+l_a_loss+','+l_c_loss+','+l_s_loss+','+p1_a_loss+','+p1_c_loss+','+p1_s_loss+','+p2_a_loss+','+p2_c_loss+','+p2_s_loss+','+k_l_loss+','+k_d_loss+','+k_u_loss+','+acc_0+','+acc_1+','+acc_2+','+acc_t+'\n')
            self.csv_file.flush()

    def log(self, text):
        ''' Write the text to log file then print it.

        Args:
            text(string): text to log
        '''
        self.log_file.write(text+'\n')
        self.log_file.flush()
        print(text)

    def add_point(self, x = None, y = None,z = None,m=None,ls=None,ps1=None,ps2=None,allo=None,cllo=None,sllo=None,ap1lo=None,cp1lo=None,sp1lo=None,ap2lo=None,cp2lo=None,sp2lo=None,k_l_loss=None,k_d_loss=None,k_u_loss=None,acc0=None,acc1=None,acc2=None,acct=None):
        ''' Add a point to the plot

        Args:
            x (Number): x coordinate value
            y (Number): y coordinate value
        '''
        if x is not None and y is not None:
            self.xs.append(x)
            self.ys.append(y)
        else:
            raise ValueError('x and y should not be None.')

        # If csv_path is not None then write x and y to file
        if self.csv_path is not None:
            self.csv_file.write(str(x)+','+str(y)+','+str(z)+','+str(m)+','+str(ls)+','+str(ps1)+','+str(ps2)+','+str(allo)+','+str(cllo)+','+str(sllo)+','+str(ap1lo)+','+str(cp1lo)+','+str(sp1lo)+','+str(ap2lo)+','+str(cp2lo)+','+str(sp2lo)+','+str(k_l_loss)+','+str(k_d_loss)+','+str(k_u_loss)+','+str(acc0)+','+str(acc1)+','+str(acc2)+','+str(acct)+'\n')
            self.csv_file.flush()

    def make_plot(self, save_path = ''):
        ''' Make plot using all stored points

        Args:
            save_path (string): where to store the plot
        '''
        fig, ax = plt.subplots()
        ax.plot(self.xs, self.ys, label=self.legend)
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel)
        ax.legend()
        ax.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)

    def close_file(self):
        ''' Close the created file objects
        '''
        if self.log_path is not None:
            self.log_file.close()
        if self.csv_path is not None:
            self.csv_file.close()

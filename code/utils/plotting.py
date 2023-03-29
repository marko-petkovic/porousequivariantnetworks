import numpy as np
import matplotlib.pyplot as plt
import torch

def moving_average(a, n=10):
    '''
    calculates moving average of sequence a with a window size of n
    '''
    
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n  


def plot_loss(model, train_loss, test_loss, len_train, len_test, epochs, ymax=12, n = 10, alpha=.3):
    '''
    plots train and test loss
    '''
    
    ma_train = moving_average(train_loss, 2*n)
    ma_test = moving_average(test_loss, n)
    
    ma_train = np.concatenate([np.array((2*n-1)*[ma_train[0]]), ma_train])
    ma_test = np.concatenate([np.array((n-1)*[ma_test[0]]), ma_test])
    
    tr, = plt.plot(np.arange(len_train*epochs)/len_train, train_loss, alpha=alpha)
    te, = plt.plot(np.arange(len_test*epochs)/len_test, test_loss, alpha=alpha)
    
    plt.plot(np.arange(len_train*epochs)/len_train, ma_train, label='train loss', color=tr.get_color())
    plt.plot(np.arange(len_test*epochs)/len_test, ma_test, label='test loss', color=te.get_color())
    
    plt.plot()
    plt.ylim(0,ymax)
    plt.xlim(0, epochs)
    plt.xlabel('epoch')
    plt.ylabel(f"{str(model.criterion).removesuffix('()')}")
    plt.legend()
    plt.title(f'{model.name} Loss');
    

def plot_true_vs_pred(trainloader, testloader, model, mn=30, mx=70,
                      y=None, y_hat=None, y_test=None, y_hat_test=None,
                      return_ys=False, plot=True, avg_hoa=True, s=10):

    if y_hat is None and y is None:
        y_hat, y = model.predict(trainloader)
    
    if y_hat_test is None and y_test is None:
        y_hat_test, y_test = model.predict(testloader)
    
    if avg_hoa:
        plot_avg_hoa(trainloader, testloader)
        
    if plot:
        _plot_true_vs_pred(y_hat, y_hat_test, y, y_test, model, mn, mx, s)

    if return_ys:
        return y, y_hat, y_test, y_hat_test

def plot_avg_hoa(trainloader, testloader):
    
    X = np.concatenate([trainloader.dataset.X, testloader.dataset.X])
    y = np.concatenate([trainloader.dataset.y, testloader.dataset.y])
    
    # in case X also includes t-atoms
    if len(X.shape) > 2:
        X = X[:,:,0]
    
    for i in np.unique(X.sum(1)):
        
        yt = y[X.sum(1)==i]
        
        plt.plot([yt.min(), yt.max()],2*[yt.mean()], label=f'{i} Al atoms')
    
    
    
    
    
    
    
def _plot_true_vs_pred(y_hat, y_hat_test, y, y_test, model, mn=30, mx=70,s=1):
    '''
    plots predicting HoA against true HoA
    '''
    plt.scatter(y, y_hat, s=s, label='train')
    plt.scatter(y_test, y_hat_test, s=s, label='test')
    plt.xlim(mn,mx)
    plt.ylim(mn,mx)
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='-', color='k', lw=1, scalex=False, scaley=False)
    plt.xlabel('True Heat of Adsorption')
    plt.ylabel('Predicted Heat of Adsorption')
    plt.legend()
    plt.title(f'True vs Predicted HoA for {model.name}');
    
    
def plot_error_dist(y_hat, y):
    plt.hist(np.abs(y_hat-y), bins=30);
    

def plot_hoa_dist(trainloader, testloader):
    '''
    plots distribution of Heat of Adsorption for train and test set
    '''
    plt.hist(trainloader.dataset.y, bins=np.arange(20,70,.5), alpha=0.6, label = 'train')
    plt.hist(testloader.dataset.y, bins=np.arange(20,70,.5), alpha=0.6, label='test')
    plt.title('Heat of adsorption distribution')
    plt.legend();
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
import seaborn as sns
import numpy as np


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def plot_classifier(model,data,targets,midpoint=0.5,pcs=None,title=None,zlabel=None,ax=None):
    '''Plots a 2d projection of the model onto the principal components.
       The data is overlayed onto the model for visualization.
    '''
    
    #Create Principal Compoenents for Visualiztion of High Dimentional Space    
    pca = PCA(n_components=2)
    if pcs is not None:
        pca.fit(pcs)
        data_transformed = pca.transform(data)
    else:
        data_transformed = pca.fit_transform(data)
    
    #Get Data Range
    xmin = np.amin(data_transformed[:,0])
    xmax = np.amax(data_transformed[:,0])
    ymin = np.amin(data_transformed[:,1])
    ymax = np.amax(data_transformed[:,1])

    #Scale Plot Range
    scaling_factor = 0.5
    xmin = xmin - (xmax - xmin)*scaling_factor/2
    xmax = xmax + (xmax - xmin)*scaling_factor/2
    ymin = ymin - (ymax - ymin)*scaling_factor/2
    ymax = ymax + (ymax - ymin)*scaling_factor/2

    #Generate Points in transformed Space
    points = 1000
    x = np.linspace(xmin,xmax,num=points)
    y = np.linspace(ymin,ymax,num=points)
    xv, yv = np.meshgrid(x,y)

    #reshape data for inverse transform
    xyt = np.concatenate((xv.reshape([xv.size,1]),yv.reshape([yv.size,1])),axis=1)
    xy = pca.inverse_transform(xyt)
    
    #predict z values for plot
    z = model.predict(xy).reshape([points,points])
    minpoint = min([min(p) for p in z])
    maxpoint = max([max(p) for p in z])
    
    #Plot Contour from Model
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    
    scaled_targets = [target/max(targets)*200 for target in targets]
    
    #Overlay Scatter Plot With Training Data
    
    #Plot Each Catagory with different Marker on Scatter Plot
    ax.scatter(data_transformed[targets==1,0],
                [1*value for value in data_transformed[targets==1,1]],
                c='k',
                cmap=plt.cm.bwr,
                marker='+',
                s=scaled_targets,
                linewidths=1.5
                )
    
    ax.scatter(data_transformed[targets==0,0],
                [1*value for value in data_transformed[targets==0,1]],
                c='k',
                cmap=plt.cm.bwr,
                marker='.',
                s=scaled_targets,
                linewidths=1.5
                )
    
    ax.grid(b=False)

    midpercent = (midpoint-minpoint)/(maxpoint-minpoint)
    centered_cmap = shiftedColorMap(plt.cm.bwr, midpoint=midpercent)
    cmap = centered_cmap
    
    if midpercent > 1:
        midpercent = 1
        cmap = plt.cm.Blues_r
    elif midpercent < 0:
        midpercent = 0
        cmap = plt.cm.Reds
    
    z = [row for row in reversed(z)]
    im = ax.imshow(z,extent=[xmin,xmax,ymin,ymax],cmap=cmap,aspect='auto')
    ax.set_aspect('auto')

    if title is not None:
        ax.set_title(title)
    
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
   
    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    if zlabel is not None:
        plt.colorbar(im, cax=cax,label=zlabel)
    else:
        plt.colorbar(im, cax=cax)    

        
def plot_model_fit(name,predicted,actual,log=False):
    if log:
        predicted = [math.log(x) for x in predicted]
        actual = [math.log(y) for y in actual]

    #mu,sigma = calculate_moments(actual,predicted)
    #r2,pval = pearsonr(predicted,actual)
    #mse = mean_squared_error(predicted,actual)
    #print(name,'R^2: ', r2,'p-val: ',pval,'MSE: ',mse)
    
    plt.scatter(predicted,actual)
    plt.title(name + ' Predicted vs. Actual')
    ax = plt.gca()
    ax.plot([-120,120], [-120,120], ls="--", c=".3")
    
    #Plot Correct Ranges
    padding_y = (max(actual) - min(actual))*0.1
    plt.ylim(min(actual)-padding_y,max(actual)+padding_y)
    
    padding_x = (max(predicted) - min(predicted))*0.1
    plt.xlim(min(predicted)-padding_x,max(predicted)+padding_x)
    
    plt.xlabel('Predicted ' + name)
    plt.ylabel('Actual ' + name)
    
    #plt.show()
    
    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
import random
import pandas as pd
import numpy as np
import math
from scipy.integrate import ode
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


#====================#
# Helper Functions   #
#--------------------#

def extractNamedColumns(data,extract_names,col_names):
    #Get Columns associated with names
    column_numbers = [col_names.index(name) for name in extract_names]
    print(column_numbers)
    #Extract Columns from Array
    data_extract = [[row[col] for col in column_numbers] for row in data]
    return data_extract

#Removes Nan Vaulues in a Series and caps the ends
def remove_NaN(x,y):
    x_sanitized = []
    y_sanitized = []
    for i,val in enumerate(y):
        if math.isnan(val):
            if i == len(y)-1:
                y_sanitized.append(y_sanitized[-1])
                x_sanitized.append(x[i])
            elif i==0:
                for j in range(len(y)):
                    if ~math.isnan(y[j]):
                        y_sanitized.append(y[j])
                        x_sanitized.append(x[i])
        else:
            x_sanitized.append(x[i])
            y_sanitized.append(y[i])
    return x_sanitized,y_sanitized


def mlode(modelDict, df, targets, specific_features,time_index='Time (h)'):
    #Create Interpolation functions for each feature
    interpFun = {}
    for feature in df.columns:
        
        if feature not in targets:
            #print(feature)
            X,y = remove_NaN(df.reset_index()[time_index].tolist(),df[feature].tolist())
            if isinstance(feature,tuple):
                feature = feature[1]
            interpFun[feature] = interp1d(X,y)
    
    print(targets)
    
    #Define the function to integrate
    def f(x,t):
        x_dot = []
        #Generate Derivatives for Each Target
        for target in targets:
            x_pred = []
            for feature in specific_features[target]:
                
                #If the Feature is dynamically changing, use the Dynamic Value
                if feature in targets:
                    x_pred= np.append(x_pred, x[targets.index(feature)])
                
                #Otherwise use a value parameterized by time
                else:
                    x_pred= np.append(x_pred, interpFun[feature](t))
            
            #Append the Predicted Derivative to the output vector
            x_dot = np.append(x_dot,modelDict[target].predict([x_pred]))
        
        #Make sure state doesn't go negative
        #eps = 10**-4
        #c = 5
        #for i,val in enumerate(x):
        #    if val < 0:
        #        x_dot[i] = -val
                
        return x_dot
    
    return f

# New Stiff Integrator
def odeintz(fun,y0,times):
    maxDelta = 10
    
    f = lambda t,x: fun(x,t)
    r = ode(f).set_integrator('dopri5',nsteps=1000,atol=1e-4)
    r.set_initial_value(y0,times[0])

    #progress bar
    #f = FloatProgress(min=0, max=max(times))
    #display(f)

    
    #Perform Integration
    x = [y0,]
    curTime = times[0]
    for nextTime in times[1:]:
        #while r.successful() and r.t < nextTime:
        while r.t < nextTime:
            if nextTime-curTime < maxDelta:
                dt = nextTime-curTime
            else:
                dt = maxDelta
                
            value = r.integrate(r.t + dt)
            curTime = r.t
            print(curTime, end='\r')
            #sleep(0.001)
            f.value = curTime
        x.append(value)
    return x

def generateTSDataSet(dataframe,features,targets,n_points=100):
    
    strains = tuple(dataframe.index.get_level_values(0).unique())
    numSamples = len(strains)
    print( 'Total Time Series in Data Set: ', numSamples )
    
    ml_df = pd.DataFrame()
    for strain in strains:
        strain_series = {}
        strain_df = dataframe.loc[(strain,slice(None)),:]
        strain_df.index = strain_df.index.get_level_values(1)
        
        #Interpolate & Smooth Each Feature & Target Then Add To Series
        for measurement in features + targets:
            #Extract Measurement
            measurement_series = strain_df[measurement].dropna()
            #print(measurement_series.index.tolist(),
            #      measurement_series.tolist())
            
            #Generate n_points interpolated points
            times = measurement_series.index.tolist()
            deltaT = (max(times) - min(times))/n_points
            
            measurement_fun = interp1d(times,
                                       measurement_series.tolist(),kind='linear')
            interpolated_measurement = measurement_fun(np.linspace(min(times),max(times),n_points))
            
            #Smooth Points
            smoothed_measurement = savgol_filter(interpolated_measurement,7,2)
            
            #If feature write out points
            if measurement in features:
                strain_series[('feature',measurement)]=smoothed_measurement
            
            #if target calculate derivative + write out points and derivative
            if measurement in targets:
                strain_series[('feature',measurement)]=smoothed_measurement
                strain_series[('target',measurement)]=np.gradient([point/deltaT for point in smoothed_measurement])
        
        #Make this more readable by breaking up into multiple lines...
        strain_df = pd.DataFrame(strain_series,
                                 index=pd.MultiIndex.from_product([[strain],np.linspace(min(times),max(times),n_points)],
                                                             names=['Strain', 'Time (h)']))
        ml_df = pd.concat([ml_df,strain_df])
        #display(ml_df)
    return ml_df

#====================#
# Plotting Functions #
#--------------------#

def plot_species_curves(modelDict, title, df, targets, specific_features, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 3),training_sets=5):
    """
    Generate a simple plot of the test and training learning curve. Returns Metrics for each predicted curve

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
    
    #Set Random Seed For training
    seed = 103
    random.seed(seed)
    
    #Create figure / plots
    fig = plt.figure(figsize=(12,16))
    #fig.set_title(title)
    #if ylim is not None:
    #    plt.ylim(*ylim)
    
    #Create subplots for each target
    ax = {}
    for i,target in enumerate(targets):
        ax[target] = plt.subplot(int(len(targets)/2)+1, 2, i+1)
    
    #Get Randomized List of all Strains
    strains = df.index.get_level_values(0).unique()
    strains = list(strains.values)
    #print(strains)
    strains = random.sample(strains, len(strains))
    
    #Pick test strain
    test_df = df.loc[(slice(strains[0],strains[0]),slice(None)),:]
    strains = strains[1:]
    
    #Create Interpolation functions for each feature in the test strain
    interpFun = {}
    #display(test_df.reset_index())
    for feature in df.columns:
        X,y = remove_NaN(test_df.reset_index()['Time (h)'].tolist(),test_df[feature].tolist())
        if isinstance(feature,tuple):
            if feature[0] == 'feature':
                feature = feature[1]
            else:
                continue

        interpFun[feature] = interp1d(X,y)  

    train_sizes = [int(len(strains)*size/training_sets)-1 for size in train_sizes]
    for i,size in enumerate(train_sizes):
        if size < 2:
            train_sizes[i] = 2
            
    #Create Fits for each training set
    fits = {}
    for training_set in range(training_sets):        
        fits[training_set] = {}

        #Generate training strain data for this training set
        training_strains = strains[0:(train_sizes[-1] + 1)]
        #print(training_strains)
        strains = strains[train_sizes[-1]:]
        endSamples = train_sizes
        #print('Strains:',strains)
        #print('End Samples',endSamples)
        sample_sets = [df.loc[(training_strains[0:endSample],slice(None)),:] for endSample in endSamples]

        #For each set size in the training set fit the model and store it
        for j,sample_set in enumerate(sample_sets):
            
            #print('Sample Set:',sample_set.index.get_level_values(0).unique().values)
            
            # Train Model
            print('Training Models for Training Set',training_set,'In Sample set',j)
            for target in targets:
                feature_indecies = [('feature', feature) for feature in specific_features[target]]
                X = sample_set[feature_indecies].values.tolist()

                #print(feature_indecies)
                #display(sample_set[feature_indecies])
                target_index = ('target',target)
                y = sample_set[target_index].values.tolist()
                modelDict[target].fit(X,y)

            print('Integrating ODEs!')
            # Integrate Given Model Test Case
            g = mlode(modelDict, test_df, targets, specific_features)
            times = test_df.reset_index()['Time (h)'].tolist()

            #Set Y0 initial condition
            appended_targets = [('feature',target) for target in targets]
            #display(test_df)
            #display(test_df[appended_targets].iloc[0])
            y0 = test_df[appended_targets].iloc[0].tolist()

            #print('times:',times)
            fit  = odeintz(g,y0,times)
            fitT = list(map(list, zip(*fit)))
            fits[training_set][train_sizes[j]] = fitT

    
    #Perform Statistics on Fits and generate plots
    colors = ['b','g','k','y','m']
    predictions = {}
    lines =[]
    labels = []
    for k,target in enumerate(targets):
        actual_data = [interpFun[target](t) for t in times]
        predictions[target] = {'actual':actual_data}
        predictions['Time'] = times
        if k == 0:
            lines.append(ax[target].plot(times,actual_data,'--', color='r')[0])
            labels.append('Actual Dynamics')
        else:
            ax[target].plot(times,actual_data,'--', color='r')
        ax[target].set_title(target)
        
        for j in range(len(sample_sets)):
            upper = []
            lower = []
            aves = []
            
            predictions[target][train_sizes[j]] = []
            for training_set in range(training_sets):
                predictions[target][train_sizes[j]].append(fits[training_set][train_sizes[j]][k])
            
            for i,time in enumerate(times):
                
                values = []
                for training_set in range(training_sets):
                    #print(training_set,train_sizes[j],i)
                    values.append(fits[training_set][train_sizes[j]][k][i])

                #Compute Statistics of Values
                #print(values)
                ave = statistics.mean(values)
                std = statistics.stdev(values)
                aves += [ave,]
                upper += [ave + std,]
                lower += [ave - std,]
                
                #print(upper)
                #print(times)
                
            #Compute upper and lower bounds for shading
            ax[target].fill_between(times, lower,upper, alpha=0.1, color=colors[j])
            if k == 0:
                lines.append(ax[target].plot(times,aves,colors[j])[0])
                labels.append(str(train_sizes[j]) + ' Strain Prediction')
            else:
                ax[target].plot(times,aves,colors[j])
            print(colors[j],train_sizes[j])
        plt.figlegend( lines, labels, loc = 'lower center', ncol=5, labelspacing=0. )       

    return predictions
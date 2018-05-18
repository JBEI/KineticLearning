
# coding: utf-8

# # A Simplified Version of the Kinetic Learning Algorithm
# 
# There was a bunch of kruft in the old kinetic learning source code.  I boiled it down into its key components and simplified the data structures.  Now the code is more managable, understandable, and extensible.
# 
# **Todo:**  
# 1. Add Smoothing to Data Augmentation as an Option.
# 2. Add a Random Seed Input
# 3. Remove Warning From Import!

# In[ ]:

import pandas as pd
from IPython.display import display
from scipy.signal import savgol_filter
import numpy as np
from tpot import TPOTRegressor
from scipy.interpolate import interp1d
from scipy.integrate import odeint,ode
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:

#Decorators
def evenly_space(fun,times):
    '''Decorate Functions that require even spacing.'''
    
    pass


# In[ ]:

def read_timeseries_data(csv_path,states,controls,impute=True,
                     time='Time',strain='Strain',augment=None,
                     est_derivative=True,smooth=False,n=None):
    '''Put DataFrame into the TSDF format.
    
    The input csv or dataframe should have a 
    column for time and ever state and control
    variable input for that time. Optional Columns are
    "Replicate" and "Strain".
    
    '''
    
    #Load Raw Data
    raw_df = pd.read_csv(csv_path)
    
    #Remove Unused Columns
    raw_df = raw_df[[strain,time] + states+controls]    

    #Standardize Index Names to Strain & Time
    raw_df.columns = [['Strain','Time']+states+controls]

    #MultiIndex Columns
    raw_df = raw_df.set_index(['Strain','Time'])
    columns = [('states',state) for state in states] + [('controls',control) for control in controls]
    raw_df.columns = pd.MultiIndex.from_tuples(columns)            
    
    #Sample num_strains Without Replacement
    if n is not None:
        strains = np.random.choice(raw_df.reset_index()['Strain'].unique(),size=n)
        raw_df = raw_df.loc[raw_df.index.get_level_values(0).isin(strains)]
    
    #Impute NaN Values using Interpolation
    if impute:
        tsdf = raw_df.groupby('Strain').apply(lambda group: group.interpolate())
    
    #Augment the data using an interpolation scheme
    if augment is not None:
        tsdf = augment_data(tsdf,n=augment)
    
    #Estimate the Derivative
    if est_derivative:
        tsdf = estimate_state_derivative(tsdf)
    
    return tsdf

def augment_data(tsdf,n=200):
    '''Augment the time series data for improved fitting.
    
    The time series data points are interpolated to create
    smooth curves for each time series and fill in blank 
    values.
    '''
    
    def augment(df):
        #Find New Times
        times = df.index.get_level_values(1)
        new_times = np.linspace(min(times),max(times),n)
        
        #Build New Indecies
        strain_name = set(df.index.get_level_values(0))
        new_indecies = pd.MultiIndex.from_product([strain_name,new_times])
        
        #Reindex the Data Frame & Interpolate New Values
        df = df.reindex(df.index.union(new_indecies))
        df.index.names = ['Strain','Time']
        df = df.interpolate()
        
        #Remove Old Indecies
        df.index = df.index.droplevel(0)
        times_to_remove = set(times) - (set(times) & set(new_times))
        df = df.loc[~df.index.isin(times_to_remove)]
        return df
            
    tsdf = tsdf.groupby('Strain').apply(augment)
    return tsdf


def estimate_state_derivative(tsdf):
    '''Estimate the Derivative of the State Variables'''
    
    #Check if a vector is evenly spaced
    evenly_spaced = lambda x: max(set(np.diff(x))) - min(set(np.diff(x))) < 10**-5
    
    #Find the difference between elements of evenly spaced vectors
    delta = lambda x: np.diff(x)[0]

    

    def estimate_derivative(tsdf):
        state_df = tsdf['states']
        times = state_df.index.get_level_values(1)
        diff = delta(times)
        
        #Find Derivative of evenly spaced data using the savgol filter
        savgol = lambda x: savgol_filter(x,7,2,deriv=1,delta=diff)
        
        if evenly_spaced(times):
            state_df = state_df.apply(savgol)      
        else:     
            state_df = state_df.apply(savgol_uneven)
            
        #Add Multicolumn
        state_df.columns = pd.MultiIndex.from_product([['derivatives'],state_df.columns])

        #Merge Derivatives Back
        tsdf = pd.merge(tsdf, state_df,left_index=True, right_index=True,how='left')

        return tsdf
    
        
    tsdf = tsdf.groupby('Strain').apply(estimate_derivative)
    return tsdf


#Reconstruct the curve using the derivative (Check that derivative Estimates are Close...)
def check_derivative(tsdf):
    '''Check the Derivative Estimates to Make sure they are good.'''
    
    #First Integrate The Derivative of Each Curve Starting at the initial condition
    
    for name,strain_df in tsdf.groupby('Strain'):
        #display(strain_df['derivatives'].tail)
        times = strain_df.index.get_level_values(1)
        dx_df = strain_df['derivatives'].apply(lambda y: interp1d(times,y,fill_value='extrapolate'))
        dx = lambda y,t: dx_df.apply(lambda x: x(t)).values
        x0 = strain_df['states'].iloc[0].values
        
        #Solve Differential Equation
        result = odeint(dx,x0,times)
        trajectory_df = pd.DataFrame(result,columns=strain_df['states'].columns)
        trajectory_df['Time'] = times
        trajectory_df = trajectory_df.set_index('Time')
        
        for column in strain_df['states'].columns:
            plt.figure()
            ax = plt.gca()
            strain_df['states'].reset_index().plot(x='Time',y=column,ax=ax)
            trajectory_df.plot(y=column,ax=ax)
            plt.show()


# In[ ]:

class dynamic_model(object):
    '''A MultiOutput Dynamic Model created from TPOT'''
    
    def __init__(self,tsdf):
        self.tsdf = tsdf

    
    def search(self,generations=50,population_size=30):
        '''Find the best model that fits the data with TPOT.'''
        
        X = self.tsdf[['states','controls']].values
        
        def fit_single_output(row):
            tpot = TPOTRegressor(generations=generations, population_size=population_size, verbosity=2,n_jobs=1)
            fit_model = tpot.fit(X,row).fitted_pipeline_
            return fit_model
    
        self.model_df = self.tsdf['derivatives'].apply(fit_single_output).to_frame()
        display(self.model_df)

    def fit(self,tsdf):
        '''Fit the Dynamical System Model.
        
        Fit the dynamical system model and
        return the map f.
        '''
        
        #update the data frame
        self.tsdf = tsdf
        X = self.tsdf[['states','controls']].values
        
        #Fit the dataframe data to existing models
        #self.model_df.apply(lambda model: print(model),axis=1)
        self.model_df = self.model_df.apply(lambda model: model[0].fit(X,self.tsdf['derivatives'][model.name]),axis=1)
    
    
    def predict(self,X):
        '''Return a Prediction'''
        y = self.model_df.apply(lambda model: model[0].predict(X.reshape(1,-1)),axis=1).values.reshape(-1,)
        return y 
    
    
    def fit_report(self):
        '''Report the Quality of the Fit in Plots'''
        #Calculate The Error Distribution, Broken down by Fit
        
        
        pass


# In[ ]:

# New Stiff Integrator
def odeintz(fun,y0,times,tolerance=1e-4):
    maxDelta = 10
    
    f = lambda t,x: fun(x,t)
    r = ode(f).set_integrator('dop853',nsteps=1000,atol=1e-4)
    r.set_initial_value(y0,times[0])

    #progress bar
    #f = FloatProgress(min=0, max=max(times))
    #display(f)

    #Perform Integration
    x = [y0,]
    curTime = times[0]
    for nextTime in times[1:]:
        #print(r.t)
        #while r.successful() and r.t < nextTime:
        while r.t < nextTime:
            if nextTime-curTime < maxDelta:
                dt = nextTime-curTime
            else:
                dt = maxDelta
                
            value = r.integrate(r.t + dt)
            curTime = r.t
            #print(curTime, end='\r')
            #sleep(0.001)
            f.value = curTime
        x.append(value)
    return x


# In[ ]:

def learn_dynamics(df,generations=50,population_size=30):
    '''Find system dynamics Time Series Data.
    
    Take in a Data Frame containing time series data 
    and use that to find the dynamics x_dot = f(x,u).
    '''    
    
    #Fit Model
    model = dynamic_model(df)
    model.search(generations=generations,population_size=population_size)
    
    return model


def simulate_dynamics(model,strain_df,time_points=None,tolerance=1e-4):
    '''Use Learned Dynamics to Generate a Simulated Trajectory in the State Space'''
    display(strain_df)
    times = strain_df.index.get_level_values(1)
    
    #Get Controls as a Function of Time Using Interpolations
    u_df = strain_df['controls'].apply(lambda y: interp1d(times,y,fill_value='extrapolate'))
    u = lambda t: u_df.apply(lambda x: x(t)).values

    #Get Initial Conditions from the Strain Data Frame
    x0 = strain_df['states'].iloc[0].values
    
    #Solve Differential Equation For Same Time Points
    #f = lambda x,t: (model.predict(np.concatenate([x, u(t)])),print(t))[0]
    f = lambda x,t: model.predict(np.concatenate([x, u(t)]))
    
    #Return DataFrame with Predicted Trajectories (Use Integrator with Sufficiently Low tolerances...)
    sol = odeintz(f,x0,times,tolerance=tolerance)
    #sol = odeint(f,x0,times,atol=5*10**-4,rtol=10**-6)
    trajectory_df = pd.DataFrame(sol,columns=strain_df['states'].columns)
    trajectory_df['Time'] = times
    #display(trajectory_df)
    
    return trajectory_df


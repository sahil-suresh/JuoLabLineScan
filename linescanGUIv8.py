# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:46:10 2022

@author: Sahil Suresh

Credits: Eli Billauer for the Peak Detection function/formula

"""

import PySimpleGUI as sg
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import os.path
import os
import json
from json import (load as jsonload, dump as jsondump)
from os import path
import io
import sys
import keyboard
from skimage import draw
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets  import RectangleSelector
import matplotlib
from scipy.io import savemat
import scipy
import scipy.signal
from numpy import NaN, Inf, arange, isscalar, asarray, array
from matplotlib.pyplot import plot, scatter, show
import pandas as pd

headerlogopath = os.getcwd() + '\\HeaderLogo.png'
defaultsettings = os.getcwd() + '\\settings_file.cfg'
defaultgroups = os.getcwd() + '\\groups_file.cfg'
np.set_printoptions(threshold=sys.maxsize)

#SETTINGS SECTION/DEFAULTS
SETTINGS_FILE = path.join(path.dirname(__file__), r"{}".format(defaultsettings))
DEFAULT_SETTINGS = {'LINEWIDTH':3,'IMAGESCALE': 50, 'PEAKDETDELTA': 50,'YMIN':0,'YMAX':1000, 'AUTOSCALE': True, 'SCIPYANALYSIS':False,
                    'STANDARDIZE':False,'SCIPYMINDISTANCE': 2,'SCIPYHEIGHT': 100,'SCIPYTHRESHOLD':10}
# "Map" from the settings dictionary keys to the window's element keys
SETTINGS_KEYS_TO_ELEMENT_KEYS = {'LINEWIDTH':'linewidth','IMAGESCALE': 'imagescale', 'PEAKDETDELTA':'peakdetdelta', 'YMIN':'ymin','YMAX':'ymax',
                                 'AUTOSCALE':'autoscale', 'SCIPYANALYSIS':'scipyanalysis', 'SCIPYMINDISTANCE':'scipymindistance',
                                 'STANDARDIZE':'standardize',
                                 'SCIPYHEIGHT':'scipyheight','SCIPYTHRESHOLD':'scipythreshold'}

#SETTINGS SECTION/DEFAULTS
GROUPS_FILE = path.join(path.dirname(__file__), r"{}".format(defaultgroups))
DEFAULT_GROUPS = {'GROUPNUMBER':1,'COLORLABELS':'Enter list of group colors in order',
'GROUPLABELS':'Enter list of group labels in order'}
# "Map" from the settings dictionary keys to the window's element keys
GROUPS_KEYS_TO_ELEMENT_KEYS = {'GROUPNUMBER':'groupnumber','COLORLABELS':'colorlabels',
'GROUPLABELS':'grouplabels'}

#LOAD SETTINGS FUNCTION
def load_settings(settings_file, default_settings):
    try:
        with open(settings_file, 'r') as f:
            settings = jsonload(f)
    except Exception as e:
        #sg.popup_quick_message(f'exception {e}', 'No settings file found... will create one for you', keep_on_top=True, background_color='red', text_color='white')
        settings = default_settings
        save_settings(settings_file, settings, None)
    return settings

#SAVE SETTINGS FUNCTION
def save_settings(settings_file, settings, values):
    if values:      # if there are stuff specified by another window, fill in those values
        for key in SETTINGS_KEYS_TO_ELEMENT_KEYS:  # update window with the values read from settings file
            try:
                settings[key] = values[SETTINGS_KEYS_TO_ELEMENT_KEYS[key]]
            except Exception as e:
                print(f'Problem updating settings from window values. Key = {key}')

    with open(settings_file, 'w') as f:
        jsondump(settings, f)
        
#LOAD GROUPS FUNCTION
def load_groups(groups_file, default_groups):
    try:
        with open(groups_file, 'r') as f:
            groups = jsonload(f)
    except Exception as e:
        #sg.popup_quick_message(f'exception {e}', 'No settings file found... will create one for you', keep_on_top=True, background_color='red', text_color='white')
        groups = default_groups
        save_groups(groups_file, groups, None)
    return groups

#SAVE GROUPS FUNCTION
def save_groups(groups_file, groups, values):
    if values:      # if there are stuff specified by another window, fill in those values
        for key in GROUPS_KEYS_TO_ELEMENT_KEYS:  # update window with the values read from settings file
            try:
                groups[key] = values[GROUPS_KEYS_TO_ELEMENT_KEYS[key]]
            except Exception as e:
                print(f'Problem updating settings from window values. Key = {key}')

    with open(groups_file, 'w') as f:
        jsondump(groups, f)

# mouse callback function
def drawing_line(event,x,y,flags,param):
    global image, image_copy, r_start, c_start,img_final, linecount, wholetrace
    linewidth = settings['LINEWIDTH']
    # If left mouse button is clicked, start of line
    if (event == cv2.EVENT_LBUTTONDOWN):
        r_start = x
        c_start = y
        start = [x,y]
        cv2.imshow(filename,img_final)
        linecount = linecount + 1
        
        
    # If left mouse button is clicked, end of line; plot intensity profile
    if (event == cv2.EVENT_LBUTTONUP):
        r_end = x
        c_end = y
        img_final = cv2.line(img_final, (r_start, c_start), (r_end, c_end), (637500, 637500, 637500), linewidth)
        cv2.imshow(filename,img_final)
        if linecount == 1:    
            firsttrace = np.transpose(np.array(draw.line(r_start, c_start, r_end, c_end)))
            wholetrace = firsttrace
            window['Coordinates'].print(wholetrace)
        else:
            currenttrace = np.transpose(np.array(draw.line(r_start, c_start, r_end, c_end)))
            wholetrace = np.concatenate((wholetrace, currenttrace),axis=0)
            window['Coordinates'].print(wholetrace)


def elibillauer(data):
    ax.plot(data, 'midnightblue')
    ax.legend(['Intensity'])
    sorted_data = np.sort(data)
    yvals=np.arange(len(sorted_data))/float(len(sorted_data)-1)
    cdfline = ax2.plot(sorted_data,yvals)
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Probability')
    if settings['AUTOSCALE'] == False:
        ymin = settings['YMIN']
        ymax = settings['YMAX']
        ax.set_ylim(ymin,ymax)
    ax.set_xlabel('')
    ax.set_ylabel('Pixel Intensity')
    delta_settings = settings['PEAKDETDELTA']
    maxtab, mintab = peakdet(data,delta_settings)
    ax.scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='red')
    window['Peaks'].update(array(maxtab))
    avgpeakintensity = np.mean(array(maxtab)[:,1])
    standarderror = scipy.stats.sem(array(maxtab)[:,1])
    peakcount = len(array(maxtab)[:,1])
    peakdistance = np.diff(array(maxtab)[:,0])
    avgpeakdistance = np.mean(peakdistance)
    peakwidth = scipy.signal.peak_widths(data,array(maxtab)[:,0])
    avgpeakwidth = np.mean(peakwidth)
    baseline = data - scipy.signal.detrend(data)
    avgbaseline = np.mean(baseline)
    window['Analysis'].print('Average Peak Intensity:', avgpeakintensity)
    window['Analysis'].print('Intensity Standard Error:', standarderror)
    window['Analysis'].print('Number of Peaks:', peakcount)
    window['Analysis'].print('Average Peak Distance:', avgpeakdistance)
    window['Analysis'].print('Average Peak Width:', avgpeakwidth)
    window['Analysis'].print('Baseline Intensity:', avgbaseline)
    return avgpeakintensity, standarderror, peakcount, avgpeakdistance, cdfline
    
def scipyanalysis(data):
    ax.plot(data, 'midnightblue')
    ax.legend(['Intensity'])
    sorted_data = np.sort(data)
    yvals=np.arange(len(sorted_data))/float(len(sorted_data)-1)
    cdfline = ax2.plot(sorted_data,yvals)
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Probability')
    if settings['AUTOSCALE'] == False:
        ymin = settings['YMIN']
        ymax = settings['YMAX']
        ax.set_ylim(ymin,ymax)
    ax.set_xlabel('')
    ax.set_ylabel('Pixel Intensity')
    scipymindistance = settings['SCIPYMINDISTANCE']
    scipyheight = settings['SCIPYHEIGHT']
    scipythreshold = settings['SCIPYTHRESHOLD']
    peaks = scipy.signal.find_peaks(data, height=scipyheight, threshold=scipythreshold, distance=scipymindistance, prominence=None, width=None, wlen=None)
    ax.scatter(peaks[0], peaks[1]['peak_heights'], color='red')
    window['Peaks'].update(peaks[0])
    window['Peaks'].print(peaks[1]['peak_heights'])
    avgpeakintensity = np.mean(peaks[1]['peak_heights'])
    standarderror = scipy.stats.sem(peaks[1]['peak_heights'])
    peakcount = len(peaks[1]['peak_heights'])
    peakdistance = np.diff(peaks[0])
    avgpeakdistance = np.mean(peakdistance)
    peakwidth = scipy.signal.peak_widths(data,peaks[0])
    avgpeakwidth = np.mean(peakwidth)
    baseline = data - scipy.signal.detrend(data)
    avgbaseline = np.mean(baseline)
    window['Analysis'].print('Average Peak Intensity:', avgpeakintensity)
    window['Analysis'].print('Intensity Standard Error:', standarderror)
    window['Analysis'].print('Number of Peaks:', peakcount)
    window['Analysis'].print('Average Peak Distance:', avgpeakdistance)
    window['Analysis'].print('Average Peak Width:', avgpeakwidth)
    window['Analysis'].print('Baseline Intensity:', avgbaseline)
    return avgpeakintensity, standarderror, peakcount, avgpeakdistance, cdfline

def peakdet(v, delta, x = None):


    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_fig_agg(fig_agg):
    fig_agg.get_tk_widget().forget()
    plt.close('all')

#SETTINGS WINDOW
def create_settings_window(settings):

    def TextLabel(text): return sg.Text(text+':', justification='r', size=(20,1))

    scipybar_layout = [[TextLabel('Distance between Peaks'), sg.Spin(values=[i for i in range(1,10001)],key='scipymindistance')],
                      [TextLabel('Minimum Height of Peak'), sg.Spin(values=[i for i in range(1,10001)],key='scipyheight')],
                      [TextLabel('Minimum Threshold'), sg.Spin(values=[i for i in range(1,10001)],key='scipythreshold')]]
    elibar_layout = [[TextLabel('delta'), sg.Spin(values=[i for i in range(1,501)],key='peakdetdelta')]]
    gensetbar_layout = [
                        [sg.Checkbox('Standardize', enable_events=True, default=False, key = 'standardize')],
                        ]

    graphicstab_layout = [
                   [TextLabel('Image Scale'), sg.Spin(values=[i for i in range(1,101)],key='imagescale')],
                   [TextLabel('Line Width'), sg.Spin(values=[i for i in range(1,101)],key='linewidth')],
                   [TextLabel('Y-Axis Range'), sg.Spin(values=[i for i in range(-50,10001)],key='ymin'),sg.Spin(values=[i for i in range(-50,10001)],key='ymax'), sg.Checkbox('Autoscale', default=False, key = 'autoscale')]
                   ]
    analysistab_layout = [
                   [sg.Checkbox('SciPy', enable_events=True, default=False, key = 'scipyanalysis')],
                   [sg.Frame(layout = elibar_layout,title='Eli Billauer Analysis', relief=sg.RELIEF_SUNKEN, tooltip='Eli Billauer PeakDet Formula',visible = True, key = 'elibarframe')],
                   [sg.Frame(layout = scipybar_layout,title='Scipy Analysis', relief=sg.RELIEF_SUNKEN, tooltip='Scipy Find Peaks Formula',visible = False, key = 'scipyframe')],
                   [sg.Frame(layout = gensetbar_layout,title='General Analysis Settings', relief=sg.RELIEF_SUNKEN, tooltip='General Analysis',visible = True, key = 'genanalysisframe')],
                   ]

    layout = [  [sg.Text('Settings', font='Any 15')],
                [sg.TabGroup([[sg.Tab('Graphics', graphicstab_layout, key='graphicstab')],[sg.Tab('Peak Analysis', analysistab_layout, key='analysistab')]], key='settingstab', tab_location='top', selected_title_color='blue')],
                [sg.Button('Save'), sg.Button('Exit')]  ]

    window = sg.Window('Settings', layout, keep_on_top=True, finalize=True,element_padding = (3,3.5), modal = True)
    
    for key in SETTINGS_KEYS_TO_ELEMENT_KEYS:   # update window with the values read from settings file
        try:
            window[SETTINGS_KEYS_TO_ELEMENT_KEYS[key]].update(value=settings[key])
        except Exception as e:
            print(f'Problem updating PySimpleGUI window from settings. Key = {key}')
        
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
               settingsentryflag = False
               break 
        if values['scipyanalysis'] == True:
            window['scipyframe'].update(visible=True)
            window['elibarframe'].update(visible=False)
        if values['scipyanalysis'] == False:
            window['scipyframe'].update(visible=False)
            window['elibarframe'].update(visible=True)
        if event == 'Save':
            settingsentryflag = True
            save_settings(SETTINGS_FILE, settings, values)
            sg.popup('Settings saved',keep_on_top = True)
            break
    
    window.close()
    
    return settingsentryflag

#SETTINGS WINDOW
def create_groups_window(groups):

    def TextLabel(text): return sg.Text(text+':', justification='c', size=(15,1))

    
    layout = [  [sg.Text('Group Editor', font='Any 15')],
                [TextLabel('Group Number'), sg.Spin(values=[i for i in range(1,101)],key='groupnumber')],
                [TextLabel('Group Labels'), sg.Input(key='grouplabels')],
                [TextLabel('Color Labels'), sg.Input(key='colorlabels')],
                [sg.Listbox(values=batchscandict, size=(60, 10), key="items",enable_events= True)],
                [sg.Button('Initialize'), sg.Button('Delete'), sg.Button('Reset'), sg.Button('Exit')]  ]

    window = sg.Window('Group Editor', layout, keep_on_top=True, finalize=True,element_padding = (3,3), modal = True)
    
    for key in GROUPS_KEYS_TO_ELEMENT_KEYS:   # update window with the values read from settings file
        try:
            window[GROUPS_KEYS_TO_ELEMENT_KEYS[key]].update(value=groups[key])
        except Exception as e:
            print(f'Problem updating PySimpleGUI window from group editor. Key = {key}')
        
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
               groupsentryflag = False
               break 
        if event == 'Initialize':
            global groupsinitializecheckpoint
            groupsinitializecheckpoint = True
            groupsentryflag = True
            save_groups(GROUPS_FILE, groups, values)
            groupnum = groups['GROUPNUMBER']
            grouplabels = list(groups['GROUPLABELS'].split(','))
            for x in range(groupnum):
                key = str(grouplabels[x])+'intensity'+str(x)
                key2 = str(grouplabels[x])+'peakcount'+str(x)
                key3 = str(grouplabels[x])+'rawtracedata'+str(x)
                key4 = str(grouplabels[x])+'tracecoordinates'+str(x)
                key5 = str(grouplabels[x])+'peakcoordinates'+str(x)
                batchscandict.setdefault(key,[])
                batchscandict.setdefault(key2,[])
                batchscandict.setdefault(key3,[])
                batchscandict.setdefault(key4,[])
                batchscandict.setdefault(key5,[])
            sg.popup('Groups initialized',keep_on_top = True)
            window['items'].update(batchscandict)
        if event == 'Reset':
            batchscandict.clear()
            sg.popup('Groups reset',keep_on_top = True)
            window['items'].update(batchscandict)
    
    window.close()
    
    return groupsentryflag


def main_window():
    sg.theme('Reddit')
    
        
        
    menu_def = [ ['File', ['Select Folder','Select Image','Export...']],
                 ['Settings',['User Preferences']],
                 ['Tools',['Edit Groups', 'ROI Scan']],
                 ['Help', ['About...','User Guide'] ]]
        
        
    FluorescenceTracetab1 = [[sg.Canvas(key="figCanvas")]]
    CDFTracetab2 = [[sg.Canvas(key="figCanvas2")]]
    Avgtab2 = [[sg.Canvas(key="figCanvas3")]]
    
    graph_group_layout = [[sg.Tab('Peak Fluorescence Intensity', FluorescenceTracetab1,font='Courier 15', key='TAB1')],
                        [sg.Tab('CDF Traces', CDFTracetab2, font='Courier 15',key='TAB2')],
                        [sg.Tab('Average Comparison', Avgtab2, font='Courier 15',key='TAB3')]]
    graph_frame_layout = [[sg.TabGroup(graph_group_layout)]]
    
    function_layout = [[sg.Button(button_text = 'Single Line Scan', key = 'LINE_SCAN'),sg.Button(button_text = 'Batch Line Scan', key = 'BATCH_SCAN'),sg.Button(button_text = 'Group Analysis', key = 'GROUP_ANALYSIS'),sg.Button('Restore',key='CLEAR')]]
        
    output = [[sg.Frame('Line Coordinates',[[sg.Multiline(size=(12,30),key='Coordinates',pad=(0,20))]])]]
    peaks = [[sg.Frame('Peak Coordinates',[[sg.Multiline(size=(12,30),key='Peaks',pad=(0,20))]])]]
    canvas = [[sg.Frame('Output',[[sg.Frame('Analysis Graphs',layout=graph_frame_layout, visible = True, key = 'GRAPHVISIBLE')],[sg.Multiline(size=(500,200),key='Analysis',pad=(0,20))]],size =(500,545)) ]]                  

    layout = [[sg.Menu(menu_def)],
              [sg.Image(r"{}".format(headerlogopath)),sg.Push()],
              [sg.Frame(layout = function_layout,title='Line Scan Tools', relief=sg.RELIEF_SUNKEN, tooltip='Perform Line Scan',element_justification='c')],
              [sg.Column(output, element_justification='c'),sg.Column(peaks, element_justification='c'),sg.Column(canvas, element_justification='c')],
               [sg.Exit()]]
    
    return sg.Window('Juo Lab Line Scanner', layout, element_justification='center', size= (1000,820))

window, settings = None, load_settings(SETTINGS_FILE, DEFAULT_SETTINGS )
window, groups = None, load_groups(GROUPS_FILE, DEFAULT_GROUPS )
window = main_window()
fig_agg = None
fig2_agg = None
batchscandict = {}
fig2 = matplotlib.figure.Figure(figsize=(6, 4), dpi=75)
ax2 = fig2.add_subplot(111)

while True:             # Event Loop
        
    event, values = window.read()
        
    if event == 'Select Image':
        # Read an image
        try:
            filename = sg.popup_get_file('Get file')
            img = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
            scale_percent = settings['IMAGESCALE']
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img_rescale = cv2.resize(img, dim)
            img_scaled = cv2.normalize(img_rescale, dst=None, alpha=0, beta=1000, norm_type=cv2.NORM_MINMAX)
            img_scaledintensity = cv2.normalize(img_rescale, dst=None, alpha=0, beta=1000, norm_type=cv2.NORM_MINMAX)
            img_final = img_scaled*50
            cv2.imshow(filename,img_final)
            linecount = 0
        except:
            sg.popup('Invalid File')

    if event == 'Select Folder':
        try:
            # Read an image
            folder_dir = sg.popup_get_folder('Get folder')
            images = []
            for filename in os.listdir(folder_dir):
                imgpath = os.path.join(folder_dir,filename)
                img = cv2.imread(imgpath,cv2.IMREAD_UNCHANGED)
                if img is not None:
                    images.append(imgpath)
                    scale_percent = settings['IMAGESCALE']
                    width = int(img.shape[1] * scale_percent / 100)
                    height = int(img.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    img_rescale = cv2.resize(img, dim)
                    img_scaled = cv2.normalize(img_rescale, dst=None, alpha=0, beta=1000, norm_type=cv2.NORM_MINMAX)
                    img_scaledintensity = cv2.normalize(img_rescale, dst=None, alpha=0, beta=1000, norm_type=cv2.NORM_MINMAX)
                    img_final = img_scaled*50
                    cv2.imshow(filename,img_final)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        except:
            sg.popup('Invalid directory')
                

    if event == 'User Preferences':
                      
        settingsentryflag = create_settings_window(settings)
        if settingsentryflag:
            window.close()
            window = main_window()
    else:
        print(event, values)
        
    if event == 'CLEAR':
        try:
            cv2.destroyAllWindows()
            window['Coordinates'].update('')
            window['Peaks'].update('')
            if fig_agg is not None:
                delete_fig_agg(fig_agg)
                delete_fig_agg(fig2_agg)
            fig2 = matplotlib.figure.Figure(figsize=(6, 4), dpi=75)
            ax2 = fig2.add_subplot(111)
            img = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
            scale_percent = settings['IMAGESCALE']
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img_rescale = cv2.resize(img, dim)
            img_scaled = cv2.normalize(img_rescale, dst=None, alpha=0, beta=1000, norm_type=cv2.NORM_MINMAX)
            img_scaledintensity = cv2.normalize(img_rescale, dst=None, alpha=0, beta=1000, norm_type=cv2.NORM_MINMAX)
            img_final = img_scaled*50
            cv2.namedWindow(filename)
            cv2.setMouseCallback(filename,drawing_line)
            cv2.imshow(filename,img_final)
            linecount = 0
        except:
            sg.popup('No Image Selected')
    
    if event == 'BATCH_SCAN':
        try:
            if groupsinitializecheckpoint == True:
                for filename in images:
                    while True:
                        if fig_agg is not None:
                            delete_fig_agg(fig_agg)
                            delete_fig_agg(fig2_agg)
                        fig = matplotlib.figure.Figure(figsize=(6, 4), dpi=75)
                        #fig3 = matplotlib.figure.Figure(figsize=(6, 4), dpi=75)
                        ax = fig.add_subplot(111)
                        #ax3 = fig3.add_subplot(111)
                        cv2.destroyAllWindows()
                        window['Coordinates'].update('')
                        img = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
                        width = int(img.shape[1] * scale_percent / 100)
                        height = int(img.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        img_rescale = cv2.resize(img, dim)
                        img_scaled = cv2.normalize(img_rescale, dst=None, alpha=0, beta=1000, norm_type=cv2.NORM_MINMAX)
                        img_scaledintensity = cv2.normalize(img_rescale, dst=None, alpha=0, beta=1000, norm_type=cv2.NORM_MINMAX)
                        img_final = img_scaled*50
                        cv2.namedWindow(filename)
                        cv2.setMouseCallback(filename,drawing_line)
                        cv2.imshow(filename,img_final)
                        linecount = 0
                        cv2.waitKey(0)
                        legend = filename.rfind('/')
                        legendend = filename.rfind('.')
                        legendlabel = filename[legend+1:legendend]
                        groupnum = groups['GROUPNUMBER']
                        groupkey = keyboard.read_key()
                        grouplabels = list(groups['GROUPLABELS'].split(','))
                        colorlabels = list(groups['COLORLABELS'].split(','))
                        try:
                            groupkeyint = int(groupkey)
                        except:
                            groupkeyint = -1
                        if settings['SCIPYANALYSIS'] == True:
                            if groupkeyint in range(groupnum):
                                print(wholetrace)
                                window['Coordinates'].update('')
                                window['Coordinates'].update(wholetrace)
                                data = img_scaledintensity.copy()[wholetrace[:, 1], wholetrace[:, 0]]
                                if settings['STANDARDIZE'] == True:
                                    minimumcord = min(data)
                                    data = data - minimumcord
                                window['Analysis'].print(legendlabel)
                                avgpeakintensity, standarderror, peakcount, avgpeakdistance, cdfline = scipyanalysis(data)
                                batchscandict[str(grouplabels[groupkeyint])+'rawtracedata'+groupkey].append(data.tolist())
                                batchscandict[str(grouplabels[groupkeyint])+'tracecoordinates'+groupkey].append(wholetrace.tolist())
                                batchscandict[str(grouplabels[groupkeyint])+'peakcoordinates'+groupkey].append(values['Peaks'])
                                ax2.legend(legendlabel)
                                for line in cdfline:
                                    line.set_color(color=str(colorlabels[groupkeyint]))
                                fig_agg = draw_figure(window['figCanvas'].TKCanvas, fig)
                                fig2_agg = draw_figure(window['figCanvas2'].TKCanvas, fig2)
                                window.refresh()
                                batchscandict[str(grouplabels[groupkeyint])+'intensity'+groupkey].append(avgpeakintensity)
                                batchscandict[str(grouplabels[groupkeyint])+'peakcount'+groupkey].append(peakcount)
                                window['Analysis'].print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                                break
                            elif groupkey == 'q':
                                sg.popup('Image skipped')    
                                break
                            else:
                                sg.popup('Please redraw the line')
                                continue
                        else:
                            if groupkeyint in range(groupnum):
                                print(wholetrace)
                                window['Coordinates'].update('')
                                window['Coordinates'].update(wholetrace)
                                data = img_scaledintensity.copy()[wholetrace[:, 1], wholetrace[:, 0]]
                                if settings['STANDARDIZE'] == True:
                                    minimumcord = min(data)
                                    data = data - minimumcord
                                window['Analysis'].print(legendlabel)
                                avgpeakintensity, standarderror, peakcount, avgpeakdistance, cdfline = elibillauer(data)
                                batchscandict[str(grouplabels[groupkeyint])+'rawtracedata'+groupkey].append(data.tolist())
                                batchscandict[str(grouplabels[groupkeyint])+'tracecoordinates'+groupkey].append(wholetrace.tolist())
                                batchscandict[str(grouplabels[groupkeyint])+'peakcoordinates'+groupkey].append(values['Peaks'])
                                ax2.legend(legendlabel)
                                for line in cdfline:
                                    line.set_color(color=str(colorlabels[groupkeyint]))
                                fig_agg = draw_figure(window['figCanvas'].TKCanvas, fig)
                                fig2_agg = draw_figure(window['figCanvas2'].TKCanvas, fig2)
                                window.refresh()
                                batchscandict[str(grouplabels[groupkeyint])+'intensity'+groupkey].append(avgpeakintensity)
                                batchscandict[str(grouplabels[groupkeyint])+'peakcount'+groupkey].append(peakcount)
                                window['Analysis'].print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                                break
                            elif groupkey == 'q':
                                sg.popup('Image skipped')
                                break
                            else:
                                sg.popup('Please redraw the line')
                                continue
            else:
                sg.popup('Please initialize groups first')   
            
        except:
            sg.popup('Unable to perform batch analysis')

    

    if event == 'LINE_SCAN':
        try:
            if fig_agg is not None:
                delete_fig_agg(fig_agg)
                fig2_agg.get_tk_widget().forget()
            fig = matplotlib.figure.Figure(figsize=(6, 4), dpi=75)
            fig3 = matplotlib.figure.Figure(figsize=(6, 4), dpi=75)
            ax = fig.add_subplot(111)
            ax3 = fig3.add_subplot(111)
            window['GRAPHVISIBLE'].update(visible=True)
            cv2.destroyAllWindows()
            cv2.namedWindow(filename)
            cv2.setMouseCallback(filename,drawing_line)
            cv2.imshow(filename,img_final)
            cv2.waitKey(0)
            groupnum = groups['GROUPNUMBER']
            groupkey = keyboard.read_key()
            try:
                groupkeyint = int(groupkey)
            except:
                groupkeyint = -1
            if groupkeyint in range(groupnum):
                grouplabels = list(groups['GROUPLABELS'].split(','))
                colorlabels = list(groups['COLORLABELS'].split(','))
                window['Coordinates'].update('')
                window['Coordinates'].update(wholetrace)
                data = img_scaledintensity.copy()[wholetrace[:, 1], wholetrace[:, 0]]
                if settings['STANDARDIZE'] == True:
                    minimumcord = min(data)
                    data = data - minimumcord
                legend = filename.rfind('/')
                legendend = filename.rfind('.')
                legendlabel = filename[legend+1:legendend]
                if settings['SCIPYANALYSIS'] == True:
                    window['Analysis'].print(legendlabel)
                    avgpeakintensity, standarderror, peakcount, avgpeakdistance, cdfline = scipyanalysis(data)
                    ax2.legend(legendlabel)
                    for line in cdfline:
                        line.set_color(color=str(colorlabels[groupkeyint]))
                    fig_agg = draw_figure(window['figCanvas'].TKCanvas, fig)
                    fig2_agg = draw_figure(window['fig2Canvas'].TKCanvas, fig2)
                    window['Analysis'].print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                else:
                    window['Analysis'].print(legendlabel)
                    avgpeakintensity, standarderror, peakcount, avgpeakdistance, cdfline = elibillauer(data)
                    ax2.legend(legendlabel)
                    for line in cdfline:
                        line.set_color(color=str(colorlabels[groupkeyint]))
                    fig_agg = draw_figure(window['figCanvas'].TKCanvas, fig)
                    fig2_agg = draw_figure(window['figCanvas2'].TKCanvas, fig2)
                    window['Analysis'].print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                batchscandict[str(grouplabels[groupkeyint])+'rawtracedata'+groupkey].append(data.tolist())
                batchscandict[str(grouplabels[groupkeyint])+'tracecoordinates'+groupkey].append(wholetrace.tolist())
                batchscandict[str(grouplabels[groupkeyint])+'peakcoordinates'+groupkey].append(values['Peaks'])
                batchscandict[str(grouplabels[groupkeyint])+'intensity'+groupkey].append(avgpeakintensity)
                batchscandict[str(grouplabels[groupkeyint])+'peakcount'+groupkey].append(peakcount)
            else:
                window['Coordinates'].update('')
                window['Coordinates'].update(wholetrace)
                data = img_scaledintensity.copy()[wholetrace[:, 1], wholetrace[:, 0]]
                if settings['STANDARDIZE'] == True:
                    minimumcord = min(data)
                    data = data - minimumcord
                legend = filename.rfind('/')
                legendend = filename.rfind('.')
                legendlabel = filename[legend+1:legendend]
                if settings['SCIPYANALYSIS'] == True:
                    window['Analysis'].print(legendlabel)
                    avgpeakintensity, standarderror, peakcount, avgpeakdistance, cdfline = scipyanalysis(data)
                    ax2.legend(legendlabel)
                    fig_agg = draw_figure(window['figCanvas'].TKCanvas, fig)
                    fig2_agg = draw_figure(window['figCanvas2'].TKCanvas, fig2)
                    window['Analysis'].print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                else:
                    window['Analysis'].print(legendlabel)
                    avgpeakintensity, standarderror, peakcount, avgpeakdistance, cdfline = elibillauer(data)
                    ax2.legend(legendlabel)
                    fig_agg = draw_figure(window['figCanvas'].TKCanvas, fig)
                    fig2_agg = draw_figure(window['figCanvas2'].TKCanvas, fig2)
                    window['Analysis'].print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        except:
             sg.popup('Line Scan Failed')
    else:
        print(event, values)
    
    if event == 'ROI Scan':
        try:
            cv2.destroyAllWindows()
            roi=cv2.selectROI("Make Selection", img_final)
            img_final = img_final[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            img_scaledintensity = img_scaledintensity[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            cv2.destroyAllWindows()
            cv2.imshow("Cropped Selection", img_final)
            sumbrightness = cv2.sumElems(img_final)
            avgbrightness = cv2.mean(img_final)
            legend = filename.rfind('/')
            legendend = filename.rfind('.')
            legendlabel = filename[legend+1:legendend]
            window['Analysis'].print(legendlabel) 
            window['Analysis'].print('Maximum Intensity in this Selection {}'.format(img_final.max())) 
            window['Analysis'].print('Minimum Intensity in this Selection {}'.format(img_final.min()))
            window['Analysis'].print('Average Brightness in this Selection {}'.format(avgbrightness))
            window['Analysis'].print('Cumulative Brightness in this Selection {}'.format(sumbrightness))
            window['Analysis'].print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        except:
            sg.popup('No ROI Scan created')
    else:
        print(event, values)
    
    #ABOUT DROPDOWN
    if event == 'About...':
        sg.popup('Designed by Sahil Suresh (2022) under the Juo Lab, Tufts University','Version 1.6, 14 July 2022',title = 'About') 
        #pathtest = os.getcwd()
        #sg.popup(str(pathtest))
    else:
        print(event, values)
        
    if event == 'Edit Groups':
        groupsentryflag = create_groups_window(groups)
        if groupsentryflag:
            window.close()
            window = main_window()
    else:
        print(event, values)
        
    if event == 'GROUP_ANALYSIS':
        try:
            groupavgintensities = []
            comparisonxaxis = []
            errorbars = []
            fig3 = matplotlib.figure.Figure(figsize=(6, 4), dpi=75)
            fig3_agg = None
            ax3 = fig3.add_subplot(111)
            #try:
            if fig3_agg is not None:
                fig3_agg.get_tk_widget().forget()
            if fig2_agg is not None:
                fig2_agg.get_tk_widget().forget()
            for x in range(groupnum):
                groupavgcalc = np.mean(batchscandict[str(grouplabels[x])+'intensity'+str(x)])
                error = scipy.stats.sem(batchscandict[str(grouplabels[x])+'intensity'+str(x)])
                errorbars.append(error)
                groupavgintensities.append(groupavgcalc)
                grouplabels = list(groups['GROUPLABELS'].split(','))
                comparisonxaxis = grouplabels
                window['Analysis'].print(str(grouplabels[x]), 'Average Intensity:', groupavgcalc) 
            ax3.bar(comparisonxaxis, groupavgintensities, yerr = errorbars,capsize=5, color='royalblue')
            ax3.set_ylabel('Pixel Intensity')
            ax3.set_title('Average Fluorescence Comparison')
            fig3_agg = draw_figure(window['figCanvas3'].TKCanvas, fig3)
            fig2_agg = draw_figure(window['figCanvas2'].TKCanvas, fig2)
            values = [v for k,v in batchscandict.items() if 'intensity' in k]
            if groupnum > 1:
                onewayanova = scipy.stats.f_oneway(*values)
                window['Analysis'].print(onewayanova) 
            window.refresh() 
        except:
            sg.popup('Analysis unsuccessful')
    else:
        print(event, values)
    
    if event == 'Export...':
        try:
            legend = filename.rfind('/')
            legendend = filename.rfind('.')
            legendlabel = filename[legend+1:legendend]
            legendlabel = legendlabel.replace('/','')
            legendlabel = legendlabel.replace('\\','')
            legendlabel = legendlabel.replace(':','')
            experimentname = sg.popup_get_text('Enter Experiment Name ')
            folderdirectory = sg.PopupGetFolder('Enter directory to save to.',title = 'Save to Directory')
            fig.savefig(str(folderdirectory) + '/'+ str(experimentname)+str(legendlabel)+'peaktrace.png')
            fig2.savefig(str(folderdirectory) + '/'+ str(experimentname)+str(legendlabel)+'cdftrace.png')
            # create json object from dictionary
            
            #savemat(str(experimentname)+'.mat', batchscandict, appendmat=False, format='5', long_field_names=False, do_compression=False, oned_as='row')
            json = json.dumps(batchscandict)
            
            # open file for writing, "w" 
            f = open( str(experimentname)+ "dict.json","w")
            
            # write json object to file
            f.write(json)
            
            # close file
            f.close()
        except:
            sg.popup('Export failed')
            
    if event in (sg.WIN_CLOSED, 'Exit'):
        try:
            cv2.destroyAllWindows()
            break
        except:
            sg.popup('Close all windows')
    
window.close()
#sortingAlgorithms.py
#A program to benchmark the time to run 10 runs of 5 sorting algorithms
#using random integer arrays of increasing size
# Author: Shane Austin 
##########################################################################
##########################################################################

import random
import time 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##########################################################################
#ALGORITHMS
##########################################################################
#This is the implementaion of Bubble sort
#Code adapted from
#https://learnonline.gmit.ie/pluginfile.php/313257/mod_resource/content/0/Sorting.py
##########################################################################

#Run Bubble sort with the generated array
def bubbleSort(arr):

    #set the amount of passes to perform decrementing by 1 each iteration
    for i in range(len(arr)-1, 0, -1):
        #iterate through the array for each pass 
        for j in range(0,i,1):
            #set up temp arrays to swap the elements
            if arr[j] > arr[j+1]:
                swap = arr[j]
                arr[j] = arr[j+1]
                arr[j+1] = swap

##########################################################################
#This is the implementation of Insertion Sort
#Code adapted from
#https://www.geeksforgeeks.org/insertion-sort/
##########################################################################

#Run insertion sort with the generated array
def insertionSort(arr):
    
    #iterate through the array
    for i in range(1, len(arr)):
        #set current element as the "key"
        key = arr[i]

        #comparison element set to adjacent integer to the left
        j = i-1

        #while the key is less than the comparison element in sub array 
        # move on to the next element
        while j >= 0 and key < arr[j] :
                arr[j + 1] = arr[j]
                j -= 1
        # insert key to the postion where it no longer passes the above conditon
        arr[j + 1] = key
##########################################################################
#This is the implementation of Merge Sort
#Code adapted from
#https://www.geeksforgeeks.org/merge-sort/
##########################################################################

#Run merge sort with the generated array
def mergeSort(arr):
    #Recursive base case
    if len(arr) > 1:
        #find the mid point and split the arrays into 2 equal sub arrays
        mid = len(arr)//2
        left = arr[:mid] 
        right = arr[mid:]
 
        #repeat this until all arrays are of length 1 then sort
        mergeSort(left) 
        mergeSort(right)
 
        #reset variables
        i = j = k = 0

        #append data to temp arrays based on comparisons
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
        #if left array only is remaining, append all
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        #if right array only is remaining, append all
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1

##########################################################################
#This is the implementation of Counting Sort
#Code adapted from
#https://www.geeksforgeeks.org/counting-sort/
##########################################################################

#Run counting sort with the generated array
def countingSort(arr):

    #find and set max and min values of the array
    max_k = int(max(arr))
    min_k = int(min(arr))

    #generate the count array populate with 0's
    countArr = [0 for j in range(max_k - min_k +1)]
    #generate the output array
    sortedArr = [0 for j in range(len(arr))]
 
    #iterate through the array tallying occurances of values
    for i in range(0, len(arr)):
        countArr[arr[i]-min_k] += 1
 
    #cumulative count of tallys
    for i in range(1, len(countArr)):
        countArr[i] += countArr[i-1]
    
    #append count value indexes to reference value to append to sorted array
    for i in range(len(arr)-1, -1, -1):
        sortedArr[countArr[arr[i] - min_k] - 1] = arr[i]
        countArr[arr[i] - min_k] -= 1

    #append to sorted array
    for i in range(0, len(arr)):
        arr[i] = sortedArr[i]
 
    return arr

##########################################################################
#This is the implementation of Tim Sort
#Code adapted from
#https://www.geeksforgeeks.org/timsort/
##########################################################################

#set the minimum run (sub array) size
MIN_MERGE = 32
 
#set run bounds so that min run is <= to a power of 2
def calcMinRun(n):
    r = 0
    while n >= MIN_MERGE:
        r |= n & 1
        n >>= 1
    return n + r

#adapted insetion sort carried out on the run sub arrays
def insertion(arr, left, right):
    for i in range(left + 1, right + 1):
        j = i
        while j > left and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1
 
#adapted merge sort to combine sub-arrays
def merge(arr, l, m, r):

    #length of left and right array
    lenLeft, lenRight = m - l + 1, r - m
    #new arrays to append to
    left, right = [], []

    for i in range(0, lenLeft):
        left.append(arr[l + i])
    for i in range(0, lenRight):
        right.append(arr[m + 1 + i])
 
    i, j, k = 0, 0, l

    #the usual merge sort 
    while i < lenLeft and j < lenRight:
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
 
        else:
            arr[k] = right[j]
            j += 1
 
        k += 1

    while i < lenLeft:
        arr[k] = left[i]
        k += 1
        i += 1
 
    while j < lenRight:
        arr[k] = right[j]
        k += 1
        j += 1
 

#Run Tim Sort with the genetated array
def timSort(arr):
    #Variable for array length and set minRun
    n = len(arr)
    minRun = calcMinRun(n)
     
    #run insertion sort iterating from 0 to end of array 
    #in increments of minRun
    for start in range(0, n, minRun):
        # use insertion sort to append and sort element from rest of main array
        # into sub array until min size is obtained 
        end = min(start + minRun - 1, n - 1)
        insertion(arr, start, end)

    #start to merge the sub arrays
    #starting at size 32 then doubling every iteration until it exceeds the array size
    size = minRun
    while size < n:

        #left size is set to 64 to combine 2 subarrays
        for left in range(0, n, 2 * size):

            #define the end point of left array 
            #and the start of right array
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))

            #merge the arrays using mergesort
            if mid < right:
                merge(arr, left, mid, right)
 
        size = 2 * size

##########################################################################
#BENCHMARKING
##########################################################################
arraySizes = (100,250,500,750,1000,1250,2500,3750,5000, 6250, 7500,8750,10000)
##########################################################################

#Run the time Log with the called algorithm and array size
def timeLog(algorithm, size):

    #array to contain test arrays
    testArrays = []

    #Defined to carry to 10 runs of each instance
    for _i in range(10):
        #generate random array form 1-99 for the given array size
        randomArray = [random.randint(0, 100) for i in range(size)]
        #store each array in testArrays to be called for testing
        testArrays.append(randomArray)

    #start timer
    start_time = time.time()

    #Run sorting algorithm for each of the arrays in testArrays
    for randomArray in testArrays:    

        algorithm(randomArray)

    #stop the timer
    end_time = time.time()

    #time taken = end time - start time, converted to milliseconds 
    #divide by 10 for the average
    time_elapsed = ((end_time - start_time)/10) *1000  
    #round output to 3 decimal places     
    return round(time_elapsed,3)

##########################################################################

#Run sortArray with the called Algorithm
def sortArray(algorithm):
    #array to store times
    sortTimes = []            
    #iterate through test sizes 
    for size in arraySizes:
        #start timer function and append output times
        times = timeLog(algorithm, size)
        sortTimes.append(times)
 
    return (sortTimes)

##########################################################################    
#run sortArray program for each algorithm
bubbleTime = sortArray(bubbleSort)
mergeTime = sortArray(mergeSort)
countingTime = sortArray(countingSort)
insertionTime = sortArray(insertionSort)
timTime = sortArray(timSort)

##########################################################################
#Setup dataframe output

sortAlgorithm = ["Bubble Sort", "Merge Sort", "Counting Sort", "Insertion Sort","Tim Sort"]

columns = list(map(str,arraySizes))

data = np.array([bubbleTime, mergeTime, countingTime, insertionTime, timTime])
df = pd.DataFrame(data = data, index = [sortAlgorithm], columns= [columns])
print(df)

##########################################################################
#Plot Chart

def plot():
    ax = plt.axes()

    plt.plot(arraySizes, bubbleTime, marker = 'o', label = 'Bubble')
    plt.plot(arraySizes, mergeTime, marker = 'o', label = 'Merge')
    plt.plot(arraySizes, countingTime, marker = 'o', label = 'Counting')
    plt.plot(arraySizes, insertionTime, marker = 'o', label = 'Insertion')
    plt.plot(arraySizes, timTime, marker = 'o', label = 'Tim')

    plt.xlabel("Array Size")
    plt.ylabel("Time (ms)")
    plt.title("Time Tests of 5 Sorting Algorithms plotted to log")
    plt.legend()

    plt.grid(linestyle ="dashed")
    ax.set_facecolor("lightgrey")
    plt.yscale('log')

    plt.savefig("Output Plot log.png")
    plt.show()

#plot()

package com.company;

import java.util.ArrayList;
import java.util.Collections;

public class UTCombination {
    static ArrayList<ArrayList<String>> temp = new ArrayList<ArrayList<String>>();
    static ArrayList<ArrayList<String>> temp2 = new ArrayList<ArrayList<String>>();
    static void combinationUtil(String arr[], String data[], int start,
                                int end, int index, int r)
    {
        // Current combination is ready to be printed, print it
        if (index == r) {
            ArrayList<String> flag = new ArrayList<String>();
            for (int j = 0; j < r; j++){
//                System.out.print(data[j] + " ");
                flag.add(data[j]);
            }
            Collections.shuffle(flag);
            temp.add(flag);
//            System.out.println("");
            return;
        }

        // replace index with all possible elements. The condition
        // "end-i+1 >= r-index" makes sure that including one element
        // at index will make a combination with remaining elements
        // at remaining positions
        for (int i=start; i<=end && end-i+1 >= r-index; i++)
        {
            data[index] = arr[i];
            combinationUtil(arr, data, i+1, end, index+1, r);
        }
    }

    // The main function that prints all combinations of size r
    // in arr[] of size n. This function mainly uses combinationUtil()
    static void printCombination(String arr[], int n, int r)
    {
        // A temporary array to store all combination one by one
        String data[]=new String[r];

        // Print all combination using temprary array 'data[]'
        combinationUtil(arr, data, 0, n-1, 0, r);
    }

    /*Driver function to check for above function*/
    public static void main (String[] args) {
        String arr[] = {"Coke.jpg", "Grape Juice.jpg", "Orange Juice.jpg",
                "Cereal.jpg", "Apple.jpg", "Orange.jpg", "Crackers.jpg", "Potato Chips.jpg",
                "Onion.jpg", "Corn.jpg",
                "Jelly.jpg", "Bread.jpg"};
        int r = 3;
        int n = arr.length;
        printCombination(arr, n, r);

        for(int i = 0; i < temp.size(); i++){
            ArrayList<String> current = new ArrayList<String>(temp.get(i));
            for(int t = 0; t < 12; t++){
                current.add(0, arr[t]);
//                System.out.println(current.get(0) + "  " + current.get(1) + "  " + current.get(2) + "  " + current.get(3));
                ArrayList<String> currently = new ArrayList<String>(current);
                temp2.add(currently);

                current.remove(0);
            }
        }

        System.out.println(temp2.size());


        for(int i = 0; i < temp2.size(); i++){
            for(int t = 0; t < temp2.get(i).size(); t++){
                if(t != temp2.get(i).size() - 1){
                    System.out.print(temp2.get(i).get(t) + ",");
                }
                else{
                    System.out.print(temp2.get(i).get(t));
                }
            }
            System.out.println();
        }

        Collections.shuffle(temp2);

        for(int i = 0; i < temp2.size(); i++){
            for(int t = 0; t < temp2.get(i).size(); t++){
                if(t != temp2.get(i).size() - 1){
                    System.out.print(temp2.get(i).get(t) + ",");
                }
                else{
                    System.out.print(temp2.get(i).get(t));
                }
            }
            System.out.println();
        }

        Collections.shuffle(temp2);

        for(int i = 0; i < temp2.size(); i++){
            for(int t = 0; t < temp2.get(i).size(); t++){
                if(t != temp2.get(i).size() - 1){
                    System.out.print(temp2.get(i).get(t) + ",");
                }
                else{
                    System.out.print(temp2.get(i).get(t));
                }
            }
            System.out.println();
        }

        Collections.shuffle(temp2);

        for(int i = 0; i < temp2.size(); i++){
            for(int t = 0; t < temp2.get(i).size(); t++){
                if(t != temp2.get(i).size() - 1){
                    System.out.print(temp2.get(i).get(t) + ",");
                }
                else{
                    System.out.print(temp2.get(i).get(t));
                }
            }
            System.out.println();
        }

    }
}

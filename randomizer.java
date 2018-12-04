package com.company;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;

public class randomizer {
    public static void main(String[] args) throws IOException {
        ArrayList<String> items = new ArrayList<String>(Arrays.asList("Apple.thumb", "Avacado.thumb", "Bag.thumb", "Basket.thumb", "Beer.thumb", "Biscuits.thumb", "Black Pepper.thumb",
                "Bowl.thumb", "Bread.thumb", "Candy Bar.thumb", "Cereal.thumb", "Chocolate Milk.thumb", "Chocolate Syrup.thumb", "Coconut Milk.thumb",
                "Coke.thumb", "Corn.thumb", "Crackers.thumb", "Cup.thumb", "Dish.thumb", "Eggs.thumb", "Fork.thumb", "French Fries.thumb", "Grape Jelly.thumb",
                "Grape Juice.thumb", "Green Tea.thumb", "Gum.thumb", "Hair Spray.thumb", "Kiwi.thumb", "Knife.thumb", "Licorice.thumb", "Macaroni.thumb",
                "Milk.thumb", "Nuts.thumb", "Onion.thumb", "Orange Juice.thumb", "Orange.thumb", "Paprika.thumb", "Pear.thumb", "Potato Chips.thumb",
                "Potato.thumb", "Pretzel.thumb", "Radish.thumb", "Rice.thumb", "Salt.thumb", "Sausage.thumb", "Smoothie.thumb", "Spoon.thumb", "Toilet Paper.thumb",
                "Tomato Paste.thumb", "Tray.thumb", "Water Bottle.thumb"));

        ArrayList<String> cutDown = new ArrayList<String>();

        ArrayList<ArrayList<String>> one = new ArrayList<ArrayList<String>>();

        ArrayList<String> temp = new ArrayList<String>();

        Collections.shuffle(items);

        for(int i =0; i < 20; i++){
            cutDown.add(items.get(i));
        }


        for(int i  = 0; i < cutDown.size(); i++){
            ArrayList<String> cutDownClone = new ArrayList<String>(cutDown);
            temp.add(cutDownClone.get(i));
            cutDownClone.remove(i);
            for(int h = 0; h < cutDownClone.size(); h++){
                ArrayList<String> cutDownClone2 = new ArrayList<String>(cutDownClone);
                temp.add(cutDownClone2.get(h));
                cutDownClone2.remove(h);

                for(int e = 0; e < cutDownClone2.size(); e++){
                    ArrayList<String> cutDownClone3 = new ArrayList<String>(cutDownClone2);
                    temp.add(cutDownClone3.get(e));
                    cutDownClone3.remove(e);
                    for(int o = 0; o < cutDownClone3.size(); o++){
                        ArrayList<String> cutDownClone4 = new ArrayList<String>(cutDownClone3);
                        temp.add(cutDownClone4.get(o));
                        ArrayList<String> CopyTemp = new ArrayList<String>(temp);
                        one.add(CopyTemp);
                        temp.remove(3);
                    }
                    temp.remove(2);
                }
                temp.remove(1);
            }
            temp.clear();

        }

        Collections.shuffle(one);

        for(int x = 0; x < one.size(); x++){
            ArrayList<String> A = new ArrayList<String>();
            A = one.get(x);

            for(int i = 0; i < A.size()-1; i++){
                System.out.print(A.get(i) + ",");
            }
            System.out.print(A.get(A.size() - 1));
            System.out.println();
        }

        Collections.shuffle(one);

        for(int x = 0; x < one.size(); x++){
            ArrayList<String> A = new ArrayList<String>();
            A = one.get(x);

            for(int i = 0; i < A.size()-1; i++){
                System.out.print(A.get(i) + ",");
            }
            System.out.print(A.get(A.size() - 1));
            System.out.println();
        }

        Collections.shuffle(one);

        for(int x = 0; x < one.size(); x++){
            ArrayList<String> A = new ArrayList<String>();
            A = one.get(x);

            for(int i = 0; i < A.size() - 1; i++){
                System.out.print(A.get(i) + ",");
            }
            System.out.print(A.get(A.size() - 1));
            System.out.println();
        }
    }
}

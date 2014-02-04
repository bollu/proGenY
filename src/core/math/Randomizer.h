#pragma once
#include "assert.h"


template <typename T>
class Randomizer {
    public:
        //takes on default value..
        Randomizer() {}
        
        Randomizer(T begin, T end) : begin_(begin), end_(end) {}
        Randomizer(T uniValue) : begin_(uniValue), end_(uniValue) {}
       
        T getValue (float normalizedRand) {
            assert(normalizedRand >= 0 && normalizedRand <=1);

            const T delta = (end_ - begin_);
            return begin_ + delta * normalizedRand;
        };
    
    private:
        T begin_, end_;
};


#include "vector.h"
template <>
class Randomizer<vector2> {
public:
     Randomizer() {}
        
        Randomizer(vector2 begin, vector2 end) : begin_(begin), end_(end) {}
        Randomizer(vector2 uniValue) : begin_(uniValue), end_(uniValue) {}
       
        vector2 getValue (float normalizedRand) {
            assert(normalizedRand >= 0 && normalizedRand <=1);

            const vector2 delta = (end_ - begin_);

            float normalizedSplit = (rand() % 1000) / 1000.0;

            return vector2(begin_.x + normalizedSplit * delta.x, begin_.y + normalizedRand * delta.y) ; 
        };

    private:
        vector2 begin_, end_;
};

#pragma once
#include <string>
#include "vector.h"
#include "../util/mathUtil.h"
#include <iostream>


/*! Base class to represent properties

This class can be used to send and receive it's derived classes in a
generalized manner. Since the derived classes are template based,
this base class serves as a convenient mechanism to send and receive
it's derived classes. the baseProperty can then be typecasted
 into the required derived type at the caller site 
 */
class baseProperty{
protected:
	baseProperty(){};
public:
	virtual ~baseProperty(){};
};


/*! Used to store a value of any type

The property class is a generalized method to store values of various types.
The values that are stored are _not_ destroyed when the Prop is destroyed.
Hence, it is advisable to only store POD's in properties. if one wishes
to store objects created on the _Heap_, managedProp must be used that actually
manages the life cycle of the value that it holds.

*/  
template <typename T>
class Prop : public baseProperty{
	private:
		T val;
	public:

	Prop(T value) : val(value){
		//this->val = value;
	}

	Prop(const Prop<T> &other){
		this->val = other.val;
	}

	~Prop(){
	}

	/*! returns the value stored by the Property

	\return the value stored within
	*/
	virtual T *getVal(){
		return &this->val;
	}


	/*! used to set the value stored by the Property

	@param[in] val the new value to be stored
	*/
	virtual void setVal(T &val){
		this->val = val;
	}

	operator T* (){
		return this->getVal();
	} 

	


};


typedef Prop<int> iProp;
typedef Prop<float> fProp;
typedef Prop<std::string> sProp;
typedef Prop<vector2> v2Prop;
typedef Prop<util::Angle> angleProp;

class _Prop_NULL{};

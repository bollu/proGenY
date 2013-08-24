
#pragma once
#include <string>
#include "vector.h"
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
	T *getVal(){
		return &this->val;
	}


	/*! used to set the value stored by the Property

	@param[in] val the new value to be stored
	*/
	void setVal(T &val){
		this->val = val;
	}

	/*! used to set the value stored by the Property from a temporary 
		variable on the stack

	@param[in] val the new value to be stored
	*/
	void setValStack(T val){
		this->val = val;
	}




};



/*!DEPRECATED. */
template<typename T>
class ptrProp : public baseProperty{
private:
	T* val;
public:
	ptrProp(T *value){
		this->val = value;
	}

	/*! returns the object stored by the managedProp
	
	\return the object stored inside.
	*/
	T* getVal() const{
		return this->val;
	}
};

/*! DEPRECATED. used to store pointers with a controlled life cycle

managedProp automatically destroys pointers that it is constructed with.
Hence, it ensures that the pointer it is assigned to _never lives longer than itself_.
This is very important to prevent memory leaks. It can be used to wrap objects that are created on the Heap.
 */
template<typename T>
class managedProp : public baseProperty{
private:
	T *val;
public:
	managedProp(T *value){
		this->val = value;
	}

	~managedProp(){
		std::cout<<"deleted; val = "<<this->val<<std::endl;
		delete(this->val);
	};


	/*! returns the object stored by the managedProp
	
	\return the object stored inside.
	*/
	T* getVal() const{
		return this->val;
	}


	
};

/*!Property that is used to tag objects

dummyProp can be added to an Object as a sentinel.
The advantage of a dummyProp is that it consumes extra space, and 
so can be used to tag objects of interest
*/
class dummyProp : public baseProperty{
public:
	dummyProp(){};
	~dummyProp(){};
};



typedef Prop<int> iProp;
typedef Prop<float> fProp;
typedef Prop<std::string> sProp;
typedef Prop<vector2> v2Prop;

//can be used as a dummy to "TAG" certain obects.
//typedef Prop<char> dummyProp;
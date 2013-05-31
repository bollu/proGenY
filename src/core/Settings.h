#pragma once
#include "Property.h"


class Settings{
public:
	Settings(){};
	void loadSettingsFromFile(std::string filePath){};

	template<typename T>
	Prop<T>* getProp(const Hash *propertyName){
		return reinterpret_cast< Prop<T>* >( settingsMap[propertyName] );
	};


private:
	std::map<const Hash *, baseProperty*>settingsMap; 

};
#pragma once
#include "../componentSys/Property.h"
#include "Hash.h"

/*!Used to save and load settings
This class provides a uniform interface to save and load 
settings. 
*/
class Settings{
public:
	Settings();
	void loadSettingsFromFile(std::string filePath);

	/*!returns a Setting with name propertyName
	
	@param [in] propertyName the name of the setting 

	\return the setting's value  
	*/
	template<typename T>
	T* getPrimitive(const Hash *propertyName);

	/*!adds a setting that is later saved 
	@param [in] propertyName the name of the setting
	@param [in] property the value of the setting as a Prop
	*/
	void addProp(const Hash *propertyName, baseProperty* property);
	
private:
	std::map<const Hash *, baseProperty*>settingsMap; 

};


#include "Settings.hpp"
#pragma once
#include "../componentSys/Property.h"
#include "../IO/logObject.h"


/*!Used to save and load settings
This class provides a uniform interface to save and load 
settings. 
*/
class Settings{
public:
	Settings(){};
	void loadSettingsFromFile(std::string filePath){};

	/*!returns a Setting with name propertyName
	
	@param [in] propertyName the name of the setting 

	\return the setting's value  
	*/
	template<typename T>
	T* getPrimitive(const Hash *propertyName){
		auto it = settingsMap.find(propertyName);

		if( it == settingsMap.end() ){
				IO::errorLog<<"trying to get a setting that does not exist.\nSetting Name: "<<
				propertyName<<IO::flush;
		};

		Prop<T> *prop = dynamic_cast< Prop<T>* >(it->second); 

		if(prop == NULL){
			IO::errorLog<<"trying to get a setting that does not exist.\nSetting Name: "<<
			propertyName<<IO::flush;
		}
		return prop->getVal();
	};
	


	/*!adds a setting that is later saved 
	@param [in] propertyName the name of the setting
	@param [in] property the value of the setting as a Prop
	*/
	void addProp(const Hash *propertyName, baseProperty* property){

		if(this->settingsMap.find(propertyName) != settingsMap.end()){
			IO::errorLog<<"trying to add a setting twice.\nSetting Name:"<<
			propertyName<<IO::flush;
			
		}

		settingsMap[propertyName] = property;
	}
	
private:
	std::map<const Hash *, baseProperty*>settingsMap; 

};
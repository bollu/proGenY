#include "logObject.h"

template<typename T>
T* Settings::getPrimitive(const Hash *propertyName){
	auto it = settingsMap.find(propertyName);

	if(it == settingsMap.end()){
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

#pragma once

#ifndef SETTINGS_HPP_
#define SETTINGS_HPP_

#include "Settings.h"
#include "logObject.h"
#include "Hash.h"
#include <iostream>


Settings::Settings(){};
void Settings::loadSettingsFromFile(std::string filePath){};

void Settings::addProp(const Hash *propertyName, baseProperty* property){

	if(this->settingsMap.find(propertyName) != settingsMap.end()){
		IO::errorLog<<"trying to add a setting twice.\nSetting Name:"<<
		propertyName<<IO::flush;
		
	}

	settingsMap[propertyName] = property;
}

#endif
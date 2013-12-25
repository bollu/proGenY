#pragma once
#include "Settings.h"
#include "logObject.h"
#include "Hash.h"
#include <iostream>
#include <fstream>

#include <cstring>

//format:
//x'space'y
vector2 string2vector(const char *str) {
	int i = 0;
	for(; str[i] != ' '; i++) {}

	char xCoord [10];
	assert(i < 10);
	memcpy(xCoord, str, i * sizeof(char));
	xCoord[i] = '\0';

	const char *yCoord = str + i;
	
	return vector2(atoi(xCoord), atoi(yCoord));
}

Settings::Settings(){
};


Settings::Setting Settings::parseSetting(const char *serializedStr) {
	

	const char type = serializedStr[0];
	const char *data = serializedStr + 1;

	switch (type) {
		case 'b':
		if(strcmp(data, "True") == 0) {
			return Settings::Setting(new bool(true));
		}
		return Settings::Setting(new bool(false));
		break;

		case 'i':
		return Settings::Setting(new int(atoi(data)));
		break;

		case 'f':
		return Settings::Setting(new float(atof(data)));
		break;

		case 'v':
		return Settings::Setting(new vector2(string2vector(data)));
		
		break;

		default:
		IO::errorLog<<"Trying to create unknown setting type"<<IO::flush;


	}
}

void Settings::loadSettingsFromFile(std::string filePath){
	std::ifstream settingsFile(filePath);
	assert(settingsFile.good());
	
	static const int NUM_CHARS = 1024;
	char name [NUM_CHARS];
	char data [NUM_CHARS];

	while(!settingsFile.eof()) {
		std::memset(name, 0, NUM_CHARS * sizeof(char));
		std::memset(data, 0, NUM_CHARS * sizeof(char));

		//every Setting is of the format Name|(Type of Data)Data
		//bollu|i20
		//magic|f12.4
		//fullscreen|bTrue

		//getline seems to absorb the '|'. Nice
		settingsFile.getline(name, NUM_CHARS, '|');
		settingsFile.getline(data, NUM_CHARS);

		Setting setting = parseSetting(data);

		/*
		IO::infoLog<<"\nSetting: "<<name<<"|value: ";
		setting.dbgLog();
		IO::infoLog<<"\n-----";
		*/
		
		//make sure that the setting is unique
		assert(this->settings.find(Hash::getHash(name)) == this->settings.end());

		this->settings.insert(std::pair<const Hash*, Setting>(Hash::getHash(name),setting));
	}
};



void Settings::saveSettingsToFile(std::string filePath){

};


void Settings::Setting::dbgLog() {
	switch(type) {
		case Type::INT:
			IO::infoLog<<*(int *)value<<IO::flush;
			break;

		case Type::FLOAT:
			IO::infoLog<<*(float *)value<<IO::flush;
			break;

		case Type::BOOL:
			if(*(bool *)value == true) {
				IO::infoLog<<"true"<<IO::flush;
			}else{
				IO::infoLog<<"false"<<IO::flush;
			}
			break;

		case Type::VECTOR2:
			vector2 v = *(vector2 *)value; 
			IO::infoLog<<"x: "<<v.x<<" |y: "<<v.y<<IO::flush;
			break;
	}
};

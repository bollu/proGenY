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
	void saveSettingsToFile(std::string filePath);

	/*!adds a setting that is later saved 
	@param [in] propertyName the name of the setting
	@param [in] property the value of the setting as a Prop
	*/
	template <typename T>
	void addSetting(const Hash *propertyName, T setting);

	template <typename T>
	T *getSetting(const Hash *propertyName);

private:

	struct Setting {
		enum Type {
			INT = 0,
			FLOAT,
			VECTOR2,
			BOOL,
		};
		Type type;

		void *value;
	
		Setting (int *i) {
			type = Type::INT;
			value = i;
		}

		Setting (float *f) {
			type = Type::FLOAT;
			value = f;
		}

		Setting(vector2 *v) {
			type = Type::VECTOR2;
			value = v;
		}

		Setting(bool *b) {
			type = Type::BOOL;
			value = b;
		}

		void dbgLog();
	};

	std::map<const Hash *,Setting> settings; 

	Setting parseSetting(const char *serializedStr);

};

#include "Settings.hpp"
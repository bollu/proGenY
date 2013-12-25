#pragma once
#include "logObject.h"



template <typename T>
inline void Settings::addSetting(const Hash *propertyName, T value) {
	Setting setting(new T(value));

	assert(this->settings.find(propertyName) == this->settings.end());
	this->settings.insert(std::pair<const Hash*, Setting>(propertyName,setting));
};

template <>
inline float *Settings::getSetting(const Hash *propertyName) {
	assert(this->settings.find(propertyName) != this->settings.end());
	Setting &setting = this->settings.at(propertyName);

	return (float *)(setting.value);
};

template <>
inline int *Settings::getSetting(const Hash *propertyName) {
	assert(this->settings.find(propertyName) != this->settings.end());
	Setting &setting = this->settings.at(propertyName);

	return (int *)(setting.value);
};

template <>
inline bool *Settings::getSetting(const Hash *propertyName) {
	assert(this->settings.find(propertyName) != this->settings.end());
	Setting &setting = this->settings.at(propertyName);

	return (bool *)(setting.value);
};

template <>
inline vector2 *Settings::getSetting(const Hash *propertyName) {
	assert(this->settings.find(propertyName) != this->settings.end());
	Setting &setting = this->settings.at(propertyName);

	return (vector2 *)(setting.value);
};

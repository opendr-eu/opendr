/*
 * Copyright 2020-2023 OpenDR European Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MACRO_PARAMS_H
#define MACRO_PARAMS_H

#define PARAM_SET_GET(type, name, qualifier, setqualifier, getqualifier) \
  qualifier:                                                             \
  type m_##name;                                                         \
  getqualifier:                                                          \
  inline type get##name() const { return m_##name; }                     \
  setqualifier:                                                          \
  inline void set##name(type name) { m_##name = name; }

#define PARAM_SET(type, name, qualifier, setqualifier) \
  qualifier:                                           \
  type m_##name;                                       \
  setqualifier:                                        \
  inline void set##name(type name) { m_##name = name; }

#define PARAM_GET(type, name, qualifier, getqualifier) \
  qualifier:                                           \
  type m_##name;                                       \
  getqualifier:                                        \
  inline type get##name() const { return m_##name; }

#define MEMBER_PARAM_SET_GET(member, type, name, qualifier, setqualifier, getqualifier) \
  getqualifier:                                                                         \
  inline type get##name() const { return member.get##name(); }                          \
  setqualifier:                                                                         \
  inline void set##name(type name) { member.set##name(name); }

#define MEMBER_PARAM_SET(member, type, name, qualifier, setqualifier, getqualifier) \
  setqualifier:                                                                     \
  inline void set##name(type name) { member.set##name(name); }

#define MEMBER_PARAM_GET(member, type, name, qualifier, setqualifier, getqualifier) \
  getqualifier:                                                                     \
  inline type get##name() const { return member.get##name(); }

#define STRUCT_PARAM_SET_GET(member, type, name, qualifier, setqualifier, getqualifier) \
  getqualifier:                                                                         \
  inline type get##name() const { return member.name; }                                 \
  setqualifier:                                                                         \
  inline void set##name(type name) { member.name = name; }

#define STRUCT_PARAM_SET(member, type, name, qualifier, setqualifier, getqualifier) \
  setqualifier:                                                                     \
  inline void set##name(type name) { member.name = name; }

#define STRUCT_PARAM_GET(member, type, name, qualifier, setqualifier, getqualifier) \
  getqualifier:                                                                     \
  inline type get##name() const { return member.name; }

#define convertStringArgument(var, val, buf) \
  if (!strcmp(buf, #val))                    \
  var = val
#endif

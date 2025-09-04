#pragma once

#ifndef OPERATOR_EXPORT
#    if (defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined(API_EXPORT)
#        define OPERATOR_EXPORT __declspec(dllexport)
#    elif (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#        define OPERATOR_EXPORT __declspec(dllimport)
#    elif defined __GNUC__ && __GNUC__ >= 4 && (defined(API_EXPORT) || defined(__APPLE__))
#        define OPERATOR_EXPORT __attribute__((visibility("default")))
#    endif
#endif

#define CALLING_CONVENTIONS __cdecl // __stdcall

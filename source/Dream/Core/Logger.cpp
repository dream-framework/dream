//
//  Logger.cpp
//  Dream
//
//  Created by Samuel Williams on 24/01/12.
//  Copyright (c) 2012 Orion Transfer Ltd. All rights reserved.
//

#include "Logger.hpp"
#include "System.hpp"

#include <thread>
#include <sstream>
#include <unordered_map>
#include <iomanip>
#include <time.h>
#include <unistd.h>

namespace Dream
{
	namespace Core
	{
		const char * Logger::level_name(LogLevel level)
		{
			switch (level) {
			case LOG_ERROR:
				return "ERROR";

			case LOG_WARN:
				return "WARN";

			case LOG_INFO:
				return "INFO";

			case LOG_DEBUG:
				return "DEBUG";

			default:
				return "UNKNOWN";
			}
		}

		void Logger::start_session()
		{
			// Print out the current time
			time_t clock_time;
			time(&clock_time);

			struct tm current_local_time;
			localtime_r(&clock_time, &current_local_time);

			const std::size_t MAX_LENGTH = 1024;
			char date_buffer[MAX_LENGTH];
			strftime(date_buffer, MAX_LENGTH, "%c", &current_local_time);

			LogBuffer log_buffer;
			log_buffer << "Logging session started at " << date_buffer;
			log(LOG_INFO, log_buffer);
		}

		Logger::Logger() : _log_level(LOG_ALL)
		{
			SystemError::reset();
			
			_output = ::dup(STDERR_FILENO);
			SystemError::check("Opening STDERR_FILENO");
			
			_log_time.reset();

			start_session();
		}

		Logger::~Logger()
		{
			::close(_output);
		}

		void Logger::header(LogLevel level) {
			LogBuffer buffer;
			buffer << "["
			       << std::setfill(' ') << std::setw(8) << _log_time.time() << "; "
			    /* << std::setw(18) */ << thread_name() << " "
			    /* << std::setw(5) */ << level_name(level) << "] ";

			const std::string & value = buffer.str();
			::write(_output, value.data(), value.size());
		}

		void Logger::log(LogLevel level, const std::string & message)
		{
			if (level & _log_level) {
				std::lock_guard<std::mutex> lock(_lock);

				header(level);

				::write(_output, message.data(), message.size());
				::write(_output, "\n", 1);
			}
		}

		void Logger::log(LogLevel level, const std::ostream & buffer)
		{
			if (level & _log_level) {
				std::lock_guard<std::mutex> lock(_lock);

				header(level);

				const std::size_t BUFFER_SIZE = 1024;
				char output_buffer[BUFFER_SIZE];

				std::streambuf * stream = buffer.rdbuf();
				stream->pubseekpos(0);

				// Write the output atomically.
				while (1) {
					std::size_t count = stream->sgetn(output_buffer, BUFFER_SIZE);

					::write(_output, output_buffer, count);

					if (count != BUFFER_SIZE)
						break;
				}

				// Write a newline.
				::write(_output, "\n", 1);
			}
		}

		void Logger::enable(LogLevel level)
		{
			_log_level = LogLevel(_log_level | level);
		}

		void Logger::disable(LogLevel level)
		{
			_log_level = LogLevel(_log_level & ~level);
		}

		void Logger::set_thread_name(std::string name)
		{
			{
				std::lock_guard<std::mutex> lock(_lock);
				
				_thread_names[std::this_thread::get_id()] = name;
			}
		}

		// The logger state must be locked before calling this function:
		std::string Logger::thread_name() const
		{
			auto i = _thread_names.find(std::this_thread::get_id());

			if (i != _thread_names.end()) {
				return i->second;
			} else {
				return "";
			}
		}

		namespace Logging {
			Logger * console() {
				static Logger * logger = NULL;

				if (!logger) {
					logger = new Logger;
				}

				return logger;
			}
		}
	}
}


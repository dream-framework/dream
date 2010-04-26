/*
 *  Network/Server.cpp
 *  This file is part of the "Dream" project, and is licensed under the GNU GPLv3.
 *
 *  Created by Administrator on 17/11/07.
 *  Copyright 2007 Samuel Williams. All rights reserved.
 *
 */

#include "Server.h"

#include "../Core/Timer.h"
#include "../Events/Loop.h"

namespace Dream {
	namespace Network {
		
		using namespace Events;
		using boost::bind;
		
#pragma mark -
#pragma mark ServerContainer
		
		IMPLEMENT_CLASS(ServerContainer)
		
		ServerContainer::ServerContainer () : m_run(false)
		{
			m_eventLoop = Loop::klass.init();
		}
		
		ServerContainer::~ServerContainer ()
		{
			stop();
		}
		
		void ServerContainer::run ()
		{		
			std::cout << "Server container running..." << std::endl;
			
			m_eventLoop->runForever();
			
			std::cout << "Server container stopped." << std::endl;
		}
		
		REF(Loop) ServerContainer::eventLoop ()
		{
			return m_eventLoop;
		}
		
		void ServerContainer::start (REF(Server) server) {
			using namespace boost;
			
			if (!m_run) {
				m_server = server;
				
				m_run = true;
				
				std::cerr << "Starting server container..." << std::endl;
				ensure(!m_thread);
				m_thread = shared_ptr<thread> (new thread(bind(&ServerContainer::run, this)));
			}
		}
		
		void ServerContainer::stop () {
			using namespace boost;

			if (m_run) {
				std::cerr << "Stopping server container..." << std::endl;
				
				// Stop the runloop
				m_eventLoop->stop();
				m_thread->join();
			}
		}
		
#pragma mark -
#pragma mark class Server

		IMPLEMENT_CLASS(Server)
				
		Server::Server (REF(Loop) eventLoop) : m_eventLoop(eventLoop)
		{
		}
		
		Server::~Server ()
		{
			if (m_eventLoop)
			{
				foreach(REF(ServerSocket) serverSocket, m_serverSockets)
				{
					m_eventLoop->stopMonitoringFileDescriptor(serverSocket);
				}
			}
		}
		
		void Server::bindToService (const char * service, SocketType sockType)
		{
			AddressesT serverAddresses = Address::interfaceAddressesForService(service, sockType);
			
			foreach(Address & addr, serverAddresses) {
				REF(ServerSocket) serverSocket(new ServerSocket(addr));
				serverSocket->connectionCallback = bind(&Server::connectionCallbackHandler, this, _1, _2, _3, _4);

				m_serverSockets.push_back(serverSocket);

				m_eventLoop->monitorFileDescriptor(serverSocket);
			}
		}
		
#pragma mark -
#pragma mark Unit Tests
		
#ifdef ENABLE_TESTING
		
		int g_messageReceivedCount;
		class TestServerClientSocket : public ClientSocket
		{
			EXPOSE_CLASS(TestServerClientSocket)
			
			class Class : public ClientSocket::Class
			{
				EXPOSE_CLASSTYPE
			};
			
			TestServerClientSocket (const SocketHandleT & h, const Address & address) : ClientSocket(h, address)
			{
			}
			
			virtual void processEvents(Loop * eventLoop, Event events)
			{
				if (events & Events::READ_READY) {
					DynamicBuffer buf(1024, true);
					
					recv(buf);
					
					std::string incomingMessage(buf.begin(), buf.end());
					
					g_messageReceivedCount += 1;
					
					std::cerr << "Message received by " << this << " fd " << this->fileDescriptor() << " : " << incomingMessage << std::endl;
					
					eventLoop->stopMonitoringFileDescriptor(this);
				}
			}
		};
		
		IMPLEMENT_CLASS(TestServerClientSocket)
		
		class TestServer : public Server
		{
			EXPOSE_CLASS(TestServer)
			
			class Class : public Server::Class
			{
				EXPOSE_CLASSTYPE
			};
			
		protected:
			// (const SocketHandleT & h, const Address & address) : ClientSocket(h, address)
			virtual void connectionCallbackHandler (Loop * eventLoop, ServerSocket * serverSocket, const SocketHandleT & h, const Address & a)
			{
				REF(ClientSocket) clientSocket(new TestServerClientSocket(h, a));
				
				std::cerr << "Accepted connection " << clientSocket << " from " << clientSocket->remoteAddress().description();
				std::cerr << " (" << clientSocket->remoteAddress().addressFamilyName() << ")" << std::endl;
				
				eventLoop->monitorFileDescriptor(clientSocket);
			}
			
		public:
			TestServer (REF(Loop) eventLoop, const char * serviceName, SocketType socketType) : Server(eventLoop)
			{
				bindToService(serviceName, socketType);
			}
			
			virtual ~TestServer ()
			{
				
			}
		};
		
		IMPLEMENT_CLASS(TestServer)
		
		REF(TimerSource) g_timer1, g_timer2, g_timer3;
		
		static void stopTimersCallback (Loop * eventLoop, TimerSource *, Event event)
		{
			std::cerr << "Stoping connection timers..." << std::endl;
			
			g_timer1->cancel();
			g_timer2->cancel();
			g_timer3->cancel();	
		}
		
		static void stopCallback (Loop * eventLoop, TimerSource *, Event event)
		{
			std::cerr << "Stopping test" << std::endl;
			eventLoop->stop();
		}
		
		int g_messageSentCount;
		int g_addressIndex;
		AddressesT g_connectAddresses;
		static void connectCallback (Loop * eventLoop, TimerSource *, Event event)
		{
			REF(ClientSocket) testConnection(new ClientSocket);
			
			testConnection->connect(g_connectAddresses[g_addressIndex++ % g_connectAddresses.size()]);
			
			StaticBuffer buf = StaticBuffer::forCString("Hello World?", false);

			g_messageSentCount += 1;
			testConnection->send(buf);
			
			testConnection->close();
		}
		
		UNIT_TEST(Server) {
			testing("Connecting and Message Sending");
			
			REF(Loop) eventLoop = Loop::klass.init();
			REF(TestServer) server = new TestServer(eventLoop, "7979", SOCK_STREAM);
						
			g_addressIndex = 0;
			g_messageReceivedCount = 0;
			g_messageSentCount = 0;
			
			g_connectAddresses = Address::addressesForName("localhost", "7979", SOCK_STREAM);

			eventLoop->scheduleTimer(TimerSource::klass.init(stopTimersCallback, 0.4));
			eventLoop->scheduleTimer(TimerSource::klass.init(stopCallback, 0.5));
			
			eventLoop->scheduleTimer(g_timer1 = TimerSource::klass.init(connectCallback, 0.05, true));
			eventLoop->scheduleTimer(g_timer2 = TimerSource::klass.init(connectCallback, 0.1, true));
			eventLoop->scheduleTimer(g_timer3 = TimerSource::klass.init(connectCallback, 0.11, true));

			eventLoop->runForever();
			
			assertTrue(g_messageSentCount >= 1, "Messages sent");
			assertEqual(g_messageSentCount, g_messageReceivedCount, "Messages sent and received successfully");
		}
		
#endif
		
	}
}
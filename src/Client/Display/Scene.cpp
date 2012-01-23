//
//  Client/Display/Scene.cpp
//  This file is part of the "Dream" project, and is released under the MIT license.
//
//  Created by Samuel Williams on 26/06/06.
//  Copyright (c) 2006 Samuel Williams. All rights reserved.
//
//

#include "Scene.h"
#include "Context.h"

#include "../../Geometry/AlignedBox.h"

// Resource loader
#include "../../Imaging/Image.h"
#include "../../Text/Font.h"
#include "../Audio/Sound.h"
#include "../Audio/OggResource.h"

#include <numeric>
#include <algorithm>

namespace Dream
{
	namespace Client
	{
		namespace Display
		{
			void ISceneManager::render_frame_for_time (TimeT time)
			{
				REF(IScene) s = current_scene();
				
				if (s) 
					s->render_frame_for_time(time);
			}
			
	#pragma mark -
			
			REF(Resources::ILoader) SceneManager::default_resource_loader ()
			{
				REF(Resources::Loader) loader = new Resources::Loader;
				
				loader->add_loader(new Imaging::Image::Loader);
				loader->add_loader(new Client::Audio::Sound::Loader);
				loader->add_loader(new Client::Audio::OggResource::Loader);
				loader->add_loader(new Text::Font::Loader);
				
				return loader;
			}
			
			SceneManager::SceneManager (REF(IContext) display_context, REF(Loop) event_loop, REF(ILoader) resource_loader)
				: _display_context(display_context), _event_loop(event_loop), _resource_loader(resource_loader), _current_sceneIsFinished(true)
			{
				_display_context->set_delegate(this);
				
				_stopwatch.start();
			}
			
			SceneManager::~SceneManager ()
			{
				
			}
			
			void SceneManager::push_scene (REF(IScene) scene)
			{
				// Save the current scene on top of the queue.
				if (_current_scene)
					_scenes.push_front(_current_scene);
				
				
				replace_scene(scene);
			}
			
			void SceneManager::replace_scene (REF(IScene) scene)
			{
				_scenes.push_front(scene);
				
				// Clear the current scene. Any attempt to render will load up the first scene.
				_current_scene = NULL;
			}
			
			void SceneManager::append_scene (REF(IScene) scene)
			{
				_scenes.push_back(scene);
			}
			
			REF(IScene) SceneManager::current_scene ()
			{
				if (_current_scene)
					return _current_scene;
				else
					return VoidScene::shared_instance();
			}
			
			REF(IContext) SceneManager::display_context ()
			{
				return _display_context;
			}
			
			REF(Loop) SceneManager::event_loop ()
			{
				return _event_loop;
			}
			
			REF(ILoader) SceneManager::resource_loader ()
			{
				return _resource_loader;
			}
			
			void SceneManager::current_scene_is_finished ()
			{
				_current_sceneIsFinished = true;
			}
			
			void SceneManager::update_current_scene ()
			{
				_current_sceneIsFinished = false;
				
				REF(IScene) s = provide_next_scene();
				
				if (_current_scene) {
					_current_scene->will_revoke_current(this);
				}
				
				if (s) {
					_current_scene = s;
					_current_scene->will_become_current(this);
					_current_scene->did_become_current();
				} else {
					_finished_callback(this);
					_current_scene = NULL;
				}
			}
			
			REF(IScene) SceneManager::provide_next_scene ()
			{
				REF(IScene) s;
				
				if (!_scenes.empty()) {
					s = _scenes.front();
					_scenes.pop_front();
				}
				
				return s;
			}
			
			void SceneManager::render_frame_for_time(PTR(IContext) context, TimeT time)
			{
				context->make_current();
				
				_stats.begin_timer(_stopwatch.time());

				if (!_current_scene || _current_sceneIsFinished)
					update_current_scene();
				
				ISceneManager::render_frame_for_time(time);
								
				_stats.update(_stopwatch.time());
				
				if (_stats.update_count() > (60 * 20))
				{
					std::cerr << "FPS: " << _stats.updates_per_second() << std::endl;
					_stats.reset();
				}
				
				context->flush_buffers();
			}
			
			void SceneManager::process_input (PTR(IContext) context, const Input & input)
			{
				if (!process(input)) {				
					// Add the event to the thread-safe queue.
					_input_queue.process(input);
				}
			}
			
			void SceneManager::process_pending_events (IInputHandler * handler)
			{
				// Remove a block of events from the input queue and pass to the handler for processing.
				_input_queue.dequeue(handler);
			}
			
			bool SceneManager::event (const Display::EventInput & ipt)
			{
				if (ipt.event() == EventInput::EXIT)
					event_loop()->stop();
				
				return false;
			}
			
			void SceneManager::set_finished_callback (FinishedCallbackT callback)
			{
				_finished_callback = callback;
			}
			
#pragma mark -
			
			void ILayer::render_frame_for_time (IScene * scene, TimeT time) {
			
			}
			
			void ILayer::did_become_current (ISceneManager * manager, IScene * scene) {
			
			}
			
			void ILayer::will_revoke_current (ISceneManager * manager, IScene * scene) {
			
			}

#pragma mark -

			void Group::render_frame_for_time (IScene * scene, TimeT time)
			{
				for (ChildrenT::iterator i = _children.begin(); i != _children.end(); i++)
				{
					(*i)->render_frame_for_time(scene, time);
				}
			}
			
			bool Group::process (const Input & input)
			{
				bool result = false;
				
				for (ChildrenT::iterator i = _children.begin(); i != _children.end(); i++)
				{
					result |= (*i)->process(input);
				}
				
				result |= IInputHandler::process(input);
				
				return result;
			}
			
			void Group::did_become_current (ISceneManager * manager, IScene * scene)
			{
				for (ChildrenT::iterator i = _children.begin(); i != _children.end(); i++)
				{
					(*i)->did_become_current(manager, scene);
				}
			}
			
			void Group::will_revoke_current (ISceneManager * manager, IScene * scene)
			{
				for (ChildrenT::iterator i = _children.begin(); i != _children.end(); i++)
				{
					(*i)->will_revoke_current(manager, scene);
				}
			}
			
			void Group::add(PTR(ILayer) child)
			{
				_children.push_back(child);
			}
			
			void Group::remove(PTR(ILayer) child)
			{
				//_children.erase(child);
				ChildrenT::iterator pos = std::find(_children.begin(), _children.end(), child);
				
				if (pos != _children.end()) {
					_children.erase(pos);
				}
			}
			
			void Group::remove_all ()
			{
				_children.resize(0);
			}
			
#pragma mark -
			
			Scene::Scene () : _sceneManager(NULL), _first_frame(true), _startTime(0), _current_time(0)
			{
				
			}
			
			Scene::~Scene()
			{
				
			}
			
			ISceneManager * Scene::manager ()
			{
				return _sceneManager;
			}
			
			ILoader * Scene::resource_loader ()
			{
				return _sceneManager->resource_loader().get();
			}
			
			void Scene::will_become_current(ISceneManager * scene_manager)
			{
				_sceneManager = scene_manager;
				_first_frame = true;
			}
			
			void Scene::did_become_current() {
				Display::ResizeInput initial_size(_sceneManager->display_context()->size());
				process(initial_size);
				
				Group::did_become_current(_sceneManager, this);
			}
			
			void Scene::will_revoke_current(ISceneManager * scene_manager)
			{
				_sceneManager = NULL;
			}
			
			void Scene::render_frame_for_time (TimeT time)
			{
				if (_first_frame) {
					_startTime = time;
					_first_frame = false;
				}
				
				_current_time = time;
				
				Group::render_frame_for_time(this, time);
			}
			
			TimeT Scene::current_time () const
			{
				return _current_time - _startTime;
			}
			
	#pragma mark -
			
			VoidScene::VoidScene ()
			{
				
			}
			
			VoidScene::~VoidScene ()
			{
				
			}
			
			bool VoidScene::process (const Display::Input & input)
			{
				return true;
			}
			
			void VoidScene::render_frame_for_time (TimeT time)
			{
				
			}
			
			REF(VoidScene) VoidScene::shared_instance ()
			{
				static REF(VoidScene) s_voidSceneSharedInstance;
				
				if (!s_voidSceneSharedInstance)
					s_voidSceneSharedInstance = new VoidScene;
				
				return s_voidSceneSharedInstance;
			}
			
		}
	}
}

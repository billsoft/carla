// Copyright 1998-2017 Epic Games, Inc. All Rights Reserved.

#include "Carla.h"
#include "Settings/CarlaSettings.h"

#include <util/ue-header-guard-begin.h>
#include "Developer/Settings/Public/ISettingsModule.h"
#include "Developer/Settings/Public/ISettingsSection.h"
#include "Developer/Settings/Public/ISettingsContainer.h"
#include "Interfaces/IPluginManager.h"
#include "ShaderCore.h"
#if WITH_EDITOR
#include "Editor/EditorEngine.h"
#include "AssetCompilingManager.h"
#include "ShaderCompiler.h"
#include "Containers/Ticker.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#endif // WITH_EDITOR
#include <util/ue-header-guard-end.h>

#define LOCTEXT_NAMESPACE "FCarlaModule"

DEFINE_LOG_CATEGORY(LogCarla);
DEFINE_LOG_CATEGORY(LogCarlaServer);

void FCarlaModule::StartupModule()
{
	AddShaderSearchPaths();
	RegisterSettings();
	LoadChronoDll();
#if WITH_EDITOR
	RegisterAutoPlayWatcher();
#endif // WITH_EDITOR
}

#if WITH_EDITOR

extern UNREALED_API UEditorEngine* GEditor;

void FCarlaModule::RegisterAutoPlayWatcher()
{
	if (!FParse::Param(FCommandLine::Get(), TEXT("CarlaAutoPlay")))
		return;

	UE_LOG(LogCarla, Log, TEXT("CarlaAutoPlay: enabled, waiting for asset/shader compilation to finish before auto-starting Play"));

	const double StartTime = FPlatformTime::Seconds();
	// Safety net so a stuck/unexpected editor state doesn't tick forever.
	constexpr double TimeoutSeconds = 30.0 * 60.0;

	FTSTicker::GetCoreTicker().AddTicker(
		FTickerDelegate::CreateLambda([StartTime](float) -> bool
	{
		if (GEditor == nullptr)
			return true;

		// Someone already started (or queued) a session manually; nothing left to do.
		if (GEditor->IsPlayingSessionInEditor() || GEditor->GetPlaySessionRequest().IsSet())
			return false;

		if (FPlatformTime::Seconds() - StartTime > TimeoutSeconds)
		{
			UE_LOG(LogCarla, Warning, TEXT("CarlaAutoPlay: gave up after %.0f seconds without meeting the auto-play conditions"), TimeoutSeconds);
			return false;
		}

		if (GEditor->GetEditorWorldContext().World() == nullptr)
			return true;

		if (FAssetCompilingManager::Get().GetNumRemainingAssets() > 0)
			return true;

		if (GShaderCompilingManager != nullptr && GShaderCompilingManager->GetNumRemainingJobs() > 0)
			return true;

		UE_LOG(LogCarla, Log, TEXT("CarlaAutoPlay: asset/shader compilation finished, auto-starting Play"));
		GEditor->RequestPlaySession(FRequestPlaySessionParams());
		return false;
	}),
	1.0f);
}

#endif // WITH_EDITOR

void FCarlaModule::AddShaderSearchPaths()
{
	// The shader virtual path is process-global. Re-running StartupModule
	// (editor Live Coding / hot-reload) must not register it twice, as
	// AddShaderSourceDirectoryMapping asserts the path is not already mapped.
	if (AllShaderSourceDirectoryMappings().Contains(TEXT("/Plugin/Carla")))
		return;

	const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("Carla"));
	if (!Plugin.IsValid())
	{
		UE_LOG(LogCarla, Error,
			TEXT("AddShaderSearchPaths: Carla plugin not found; shaders will be unavailable."));
		return;
	}

	const FString ShadersDirectoryPath = FPaths::ConvertRelativePathToFull(
		FPaths::Combine(Plugin->GetBaseDir(), TEXT("Shaders")));
	if (!FPaths::DirectoryExists(ShadersDirectoryPath))
	{
		UE_LOG(LogCarla, Error,
			TEXT("AddShaderSearchPaths: shader directory '%s' does not exist; shaders will be unavailable."),
			*ShadersDirectoryPath);
		return;
	}

	UE_LOG(LogCarla, Log, TEXT("Carla shader directory: %s"), *ShadersDirectoryPath);
	AddShaderSourceDirectoryMapping(TEXT("/Plugin/Carla"), ShadersDirectoryPath);
}

void FCarlaModule::LoadChronoDll()
{
	#if defined(WITH_CHRONO) && PLATFORM_WINDOWS
	const FString BaseDir = FPaths::Combine(*FPaths::ProjectPluginsDir(), TEXT("Carla"));
	const FString DllDir = FPaths::Combine(*BaseDir, TEXT("CarlaDependencies"), TEXT("dll"));
	FString ChronoEngineDll = FPaths::Combine(*DllDir, TEXT("ChronoEngine.dll"));
	FString ChronoVehicleDll = FPaths::Combine(*DllDir, TEXT("ChronoEngine_vehicle.dll"));
	FString ChronoModelsDll = FPaths::Combine(*DllDir, TEXT("ChronoModels_vehicle.dll"));
	FString ChronoRobotDll = FPaths::Combine(*DllDir, TEXT("ChronoModels_robot.dll"));
	UE_LOG(LogCarla, Log, TEXT("Loading Dlls from: %s"), *DllDir);
	auto ChronoEngineHandle = FPlatformProcess::GetDllHandle(*ChronoEngineDll);
	if (!ChronoEngineHandle)
	{
		UE_LOG(LogCarla, Warning, TEXT("Error: ChronoEngine.dll could not be loaded"));
	}
	auto ChronoVehicleHandle = FPlatformProcess::GetDllHandle(*ChronoVehicleDll);
	if (!ChronoVehicleHandle)
	{
		UE_LOG(LogCarla, Warning, TEXT("Error: ChronoEngine_vehicle.dll could not be loaded"));
	}
	auto ChronoModelsHandle = FPlatformProcess::GetDllHandle(*ChronoModelsDll);
	if (!ChronoModelsHandle)
	{
		UE_LOG(LogCarla, Warning, TEXT("Error: ChronoModels_vehicle.dll could not be loaded"));
	}
	auto ChronoRobotHandle = FPlatformProcess::GetDllHandle(*ChronoRobotDll);
	if (!ChronoRobotHandle)
	{
		UE_LOG(LogCarla, Warning, TEXT("Error: ChronoModels_robot.dll could not be loaded"));
	}
	#endif
}

void FCarlaModule::ShutdownModule()
{
	if (UObjectInitialized())
	{
		UnregisterSettings();
	}
}

void FCarlaModule::RegisterSettings()
{
	// Registering some settings is just a matter of exposing the default UObject of
	// your desired class, add here all those settings you want to expose
	// to your LDs or artists.

	if (ISettingsModule* SettingsModule = FModuleManager::GetModulePtr<ISettingsModule>("Settings"))
	{
		// Create the new category
		ISettingsContainerPtr SettingsContainer = SettingsModule->GetContainer("Project");

		SettingsContainer->DescribeCategory("CARLASettings",
			LOCTEXT("RuntimeWDCategoryName", "CARLA Settings"),
			LOCTEXT("RuntimeWDCategoryDescription", "CARLA plugin settings"));

		// Register the settings
		ISettingsSectionPtr SettingsSection = SettingsModule->RegisterSettings("Project", "CARLASettings", "General",
			LOCTEXT("RuntimeGeneralSettingsName", "General"),
			LOCTEXT("RuntimeGeneralSettingsDescription", "General configuration for the CARLA plugin"),
			GetMutableDefault<UCarlaSettings>()
		);

		// Register the save handler to your settings, you might want to use it to
		// validate those or just act to settings changes.
		if (SettingsSection.IsValid())
		{
			SettingsSection->OnModified().BindRaw(this, &FCarlaModule::HandleSettingsSaved);
		}
	}
}

void FCarlaModule::UnregisterSettings()
{
	// Ensure to unregister all of your registered settings here, hot-reload would
	// otherwise yield unexpected results.

	if (ISettingsModule* SettingsModule = FModuleManager::GetModulePtr<ISettingsModule>("Settings"))
	{
		SettingsModule->UnregisterSettings("Project", "CustomSettings", "General");
	}
}

bool FCarlaModule::HandleSettingsSaved()
{
	UCarlaSettings* Settings = GetMutableDefault<UCarlaSettings>();
	bool ResaveSettings = false;

	// Put any validation code in here and resave the settings in case an invalid
	// value has been entered

	if (ResaveSettings)
	{
		Settings->SaveConfig();
	}

	return true;
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FCarlaModule, Carla)

// =============================================================================
// -- Implement carla throw_exception ------------------------------------------
// =============================================================================

#ifdef LIBCARLA_NO_EXCEPTIONS
#include <util/disable-ue4-macros.h>
#include <carla/Exception.h>
#include <util/enable-ue4-macros.h>

#include <exception>
namespace carla {

  void throw_exception(const std::exception &e) {
    UE_LOG(LogCarla, Fatal, TEXT("Exception thrown: %s"), UTF8_TO_TCHAR(e.what()));
    // It should never reach this part.
    std::terminate();
  }

} // namespace carla
#endif

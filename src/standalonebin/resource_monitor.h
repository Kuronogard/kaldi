#ifndef __RESOURCE_MONITOR_H__
    #define __RESOURCE_MONITOR_H

    #if PLATFORM_INTEL
      #include "standalonebin/resource_monitor_intel.h"
      
      typedef ResourceMonitorIntel ResourceMonitor;

    #elif PLATFORM_ARM
      #include "standalonebin/resource_monitor_ARM.h"
      
      typedef ResourceMonitorARM ResourceMonitor;
    #else
     #pragma GCC error "You must define a platform. Either PLATFORM_INTEL or PLATFORM_ARM"
    #endif
    
#endif

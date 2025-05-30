
//  ==> COPYRIGHT (C) 2022 XSENS TECHNOLOGIES OR SUBSIDIARIES WORLDWIDE <==
//  WARNING: COPYRIGHT (C) 2022 XSENS TECHNOLOGIES OR SUBSIDIARIES WORLDWIDE. ALL RIGHTS RESERVED.
//  THIS FILE AND THE SOURCE CODE IT CONTAINS (AND/OR THE BINARY CODE FILES FOUND IN THE SAME
//  FOLDER THAT CONTAINS THIS FILE) AND ALL RELATED SOFTWARE (COLLECTIVELY, "CODE") ARE SUBJECT
//  TO AN END USER LICENSE AGREEMENT ("AGREEMENT") BETWEEN XSENS AS LICENSOR AND THE AUTHORIZED
//  LICENSEE UNDER THE AGREEMENT. THE CODE MUST BE USED SOLELY WITH XSENS PRODUCTS INCORPORATED
//  INTO LICENSEE PRODUCTS IN ACCORDANCE WITH THE AGREEMENT. ANY USE, MODIFICATION, COPYING OR
//  DISTRIBUTION OF THE CODE IS STRICTLY PROHIBITED UNLESS EXPRESSLY AUTHORIZED BY THE AGREEMENT.
//  IF YOU ARE NOT AN AUTHORIZED USER OF THE CODE IN ACCORDANCE WITH THE AGREEMENT, YOU MUST STOP
//  USING OR VIEWING THE CODE NOW, REMOVE ANY COPIES OF THE CODE FROM YOUR COMPUTER AND NOTIFY
//  XSENS IMMEDIATELY BY EMAIL TO INFO@XSENS.COM. ANY COPIES OR DERIVATIVES OF THE CODE (IN WHOLE
//  OR IN PART) IN SOURCE CODE FORM THAT ARE PERMITTED BY THE AGREEMENT MUST RETAIN THE ABOVE
//  COPYRIGHT NOTICE AND THIS PARAGRAPH IN ITS ENTIRETY, AS REQUIRED BY THE AGREEMENT.
//  
//  THIS SOFTWARE CAN CONTAIN OPEN SOURCE COMPONENTS WHICH CAN BE SUBJECT TO 
//  THE FOLLOWING GENERAL PUBLIC LICENSES:
//  ==> Qt GNU LGPL version 3: http://doc.qt.io/qt-5/lgpl.html <==
//  ==> LAPACK BSD License:  http://www.netlib.org/lapack/LICENSE.txt <==
//  ==> StackWalker 3-Clause BSD License: https://github.com/JochenKalmbach/StackWalker/blob/master/LICENSE <==
//  ==> Icon Creative Commons 3.0: https://creativecommons.org/licenses/by/3.0/legalcode <==
//  

#include "xsensdeviceapi/xdaconfig.h"
#include "xsensdeviceapi/xdadll.h"
#include "xsensdeviceapi/xdainfo.h"
#include "xsensdeviceapi/xsdeviceref.h"
#include "xsensdeviceapi/xscontrollerconfig.h"
#include "xsensdeviceapi/xsaccesscontrolmode.h"
#include "xsensdeviceapi/xsalignmentframe.h"
#include "xsensdeviceapi/xscalibrateddatamode.h"
#include "xsensdeviceapi/xscallback.h"
#include "xsensdeviceapi/xscallbackplainc.h"
#include "xsensdeviceapi/xsconnectivitystate.h"
#include "xsensdeviceapi/xscoordinatesystem.h"
#include "xsensdeviceapi/xsdef.h"
#include "xsensdeviceapi/xsdeviceconfiguration.h"
#include "xsensdeviceapi/xsdeviceparameter.h"
#include "xsensdeviceapi/xsdeviceparameteridentifier.h"
#include "xsensdeviceapi/xsdeviceptr.h"
#include "xsensdeviceapi/xsdeviceptrarray.h"
#include "xsensdeviceapi/xsdevicestate.h"
#include "xsensdeviceapi/xserrormode.h"
#include "xsensdeviceapi/xsfloatformat.h"
#include "xsensdeviceapi/xsgnssplatform.h"
#include "xsensdeviceapi/xsubloxgnssplatform.h"
#include "xsensdeviceapi/xsicccommand.h"
#include "xsensdeviceapi/xsiccrepmotionresult.h"
#include "xsensdeviceapi/xsoperationalmode.h"
#include "xsensdeviceapi/xsorientationmode.h"
#include "xsensdeviceapi/xsprocessingflag.h"
#include "xsensdeviceapi/xsprotocoltype.h"
#include "xsensdeviceapi/xsrejectreason.h"
#include "xsensdeviceapi/xsscanner.h"
#include "xsensdeviceapi/xsselftestresult.h"
#include "xsensdeviceapi/xsusbhubinfo.h"
#include "xsensdeviceapi/xsdevice.h"
#include "xsensdeviceapi/xscontrol.h"
#include "xsensdeviceapi/xscontrol.h"
#include "xsensdeviceapi/xsdevice.h"

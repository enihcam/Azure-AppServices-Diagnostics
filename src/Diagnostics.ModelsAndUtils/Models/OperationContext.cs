﻿using System;
using System.Collections.Generic;
using System.Text;

namespace Diagnostics.ModelsAndUtils.Models
{
    /// <summary>
    /// Operation Context
    /// </summary>
    public class OperationContext<TResource> where TResource : IResource
    {
        /// <summary>
        /// Resource Object
        /// See <see cref="App"/> and <see cref="HostingEnvironment"/>
        /// </summary>
        public TResource Resource { get; private set; }

        /// <summary>
        /// Start Time(UTC) for data measurement
        /// </summary>
        public string StartTime { get; private set; }

        /// <summary>
        /// End Tim(UTC) for data measurement
        /// </summary>
        public string EndTime { get; private set; }

        /// <summary>
        /// sets to false when detector is called from external source (Azure portal, CLI ...)
        /// sets to true when detector is called from internal source (Applens ..)
        /// </summary>
        public bool IsInternalCall { get; private set; }

        /// <summary>
        /// Request Id
        /// </summary>
        public string RequestId { get; private set; }

        /// <summary>
        /// TimeGrain in minutes for aggregating data.
        /// </summary>
        public string TimeGrain { get; private set; }

        public OperationContext(TResource resource, string startTimeStr, string endTimeStr, bool isInternalCall, string requestId, string timeGrain = "5")
        {
            Resource = resource;
            StartTime = startTimeStr;
            EndTime = endTimeStr;
            IsInternalCall = isInternalCall;
            RequestId = requestId;
            TimeGrain = timeGrain;
        }
    }
}

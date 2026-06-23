/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <sstream>
#include "atb_speed/log.h"
#include "executor_manager.h"

namespace atb_speed {
namespace common {


int ExecutorManager::RegisterReference(aclOpExecutor *executor)
{
    std::map<aclOpExecutor *, int>::iterator it = this->executorCount_.find(executor);
    if (it != this->executorCount_.end()) {
        // The address is already tracked. Since a live executor's address can
        // never be reissued by the allocator, this is a stale entry left by a
        // previously destroyed executor whose address has been reused (ABA).
        // Discard the stale count and start fresh to avoid an inflated count
        // that would never reach zero (which would leak the executor and all of
        // its tensors).
        ATB_SPEED_LOG_WARN("Plugin Op Cache: LEAKDBG stale reference count " << it->second
            << " found for reused executor addr[" << executor << "], reset to 1 (ABA detected)");
        it->second = 1;
        return 1;
    }
    this->executorCount_[executor] = 1;
    return 1;
}

int ExecutorManager::IncreaseReference(aclOpExecutor *executor)
{
    std::map<aclOpExecutor *, int>::iterator it = this->executorCount_.find(executor);
    if (it == this->executorCount_.end()) {
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Executor addr[" << executor << "] not found in ExecutorManager, add one");
        this->executorCount_[executor] = 1;
        return 1;
    }

    int &count = it->second;
    count += 1;
    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: ExecutorManager Executor addr["
                  << executor << "] increase reference to " << count);
    return count;
}

int ExecutorManager::DecreaseReference(aclOpExecutor *executor)
{
    std::map<aclOpExecutor *, int>::iterator it = this->executorCount_.find(executor);
    if (it == this->executorCount_.end()) {
        ATB_SPEED_LOG_ERROR("Plugin Op Cache: Executor addr[" << executor << "] not found in ExecutorManager");
        return 0;
    }
    int &count = it->second;
    if (count == 1) {
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: delete Executor addr[" << executor << "]");
        this->executorCount_.erase(executor);
        return 0;
    }

    count -= 1;
    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: ExecutorManager Executor addr["
                  << executor << "] decrease reference to " << count);
    return count;
}

std::string ExecutorManager::PrintExecutorCount()
{
    std::stringstream ss;
    ss << "Plugin Op Cache: Executor Summary ";
    std::map<aclOpExecutor *, int>::iterator it;
    for (it = this->executorCount_.begin(); it != this->executorCount_.end(); it++) {
        ss << "Executor Addr[" << it->first << "] count " << it->second << " ";
    }
    return ss.str();
}

} // namespace common
} // namespace atb_speed

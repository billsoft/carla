// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include <memory>
#include <mutex>

namespace carla {

  /// A very simple atomic shared ptr with release-acquire memory order.
  /// FIXED for C++20/VS2026: Removed deprecated std::atomic_load/store for shared_ptr
  /// Replaced with std::mutex protection
  template <typename T>
  class AtomicSharedPtr {
  public:

    template <typename... Args>
    explicit AtomicSharedPtr(Args &&... args)
      : _ptr(std::forward<Args>(args)...) {}

    AtomicSharedPtr(const AtomicSharedPtr &rhs) {
        std::lock_guard<std::mutex> lock(rhs._mutex);
        _ptr = rhs._ptr;
    }

    AtomicSharedPtr(AtomicSharedPtr &&) = delete;

    void store(std::shared_ptr<T> ptr) noexcept {
      std::lock_guard<std::mutex> lock(_mutex);
      _ptr = ptr;
    }

    void reset(std::shared_ptr<T> ptr = nullptr) noexcept {
      store(ptr);
    }

    std::shared_ptr<T> load() const noexcept {
      std::lock_guard<std::mutex> lock(_mutex);
      return _ptr;
    }

    bool compare_exchange(std::shared_ptr<T> *expected, std::shared_ptr<T> desired) noexcept {
      std::lock_guard<std::mutex> lock(_mutex);
      if (_ptr == *expected) {
          _ptr = desired;
          return true;
      } else {
          *expected = _ptr;
          return false;
      }
    }

    AtomicSharedPtr &operator=(std::shared_ptr<T> ptr) noexcept {
      store(std::move(ptr));
      return *this;
    }

    AtomicSharedPtr &operator=(const AtomicSharedPtr &rhs) noexcept {
      store(rhs.load());
      return *this;
    }

    AtomicSharedPtr &operator=(AtomicSharedPtr &&) = delete;

  private:
    mutable std::mutex _mutex;
    std::shared_ptr<T> _ptr;
  };

} // namespace carla

cmake_minimum_required(VERSION 3.8)
project(robot_arm_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# depthai_ros_driver resolves the NN blob from the json's "model" block only, and
# json files get no launch substitutions -- so with zoo "path" the model_name has to
# be an absolute path. Bake this machine's install prefix in at build time rather
# than committing one: every clone then gets a path that is correct for itself.
set(BLOB_PATH "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/params/yolo8s_openvino_2022.1_5shave.blob")
configure_file(params/yolo8s.json.in ${CMAKE_CURRENT_BINARY_DIR}/yolo8s.json @ONLY)

# Install the 3D meshes (the .in template is excluded -- only the filled-in copy below ships)
install(DIRECTORY assets launch params
  DESTINATION share/${PROJECT_NAME}
  PATTERN "*.in" EXCLUDE
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/yolo8s.json
  DESTINATION share/${PROJECT_NAME}/params
)

# Install the URDF and config files
install(FILES robot.urdf config.json
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

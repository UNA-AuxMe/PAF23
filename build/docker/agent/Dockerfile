# Based on https://github.com/ll7/paf21-1/blob/master/components/carla_ros_bridge/Dockerfile

# This is commented out because of the large image download necessary to build this image otherwise.
# If the PythonAPI/carla version changes, one can comment out the Download Carla PythonAPI line below.
# Then, uncomment this line and the COPY --from=carla line to build the image.
# FROM ghcr.io/nylser/carla:leaderboard as carla
# Use this image to enable instance segmentation cameras
# FROM carlasim/carla:0.9.14 as carla

# supply the base image with an environment supporting ROS UI via x11
FROM osrf/ros:noetic-desktop-full-focal

# COPY --from=carla /home/carla/PythonAPI /opt/carla/PythonAPI

ARG USERNAME
ARG USER_UID
ARG USER_GID
ARG DEBIAN_FRONTEND=noninteractive

# install rendering dependencies for rviz / rqt
RUN apt-get update \
    && apt-get install -y -qq --no-install-recommends \
    libxext6 libx11-6 libglvnd0 libgl1 \
    libglx0 libegl1 freeglut3-dev apt-utils \
    fprintd libfprint-2-2 libpam-fprintd

# install dependencies for libgit2 and Carla PythonAPI
RUN apt-get install wget unzip

# Download Carla PythonAPI (alternative to getting it from the Carla-Image, which is commented out above)
# If the PythonAPI/Carla version changes, either update the link, or refer to the comment at the top of this file.
RUN wget https://github.com/una-auxme/paf/releases/download/v0.0.1/PythonAPI_Leaderboard-2.0.zip --quiet -O PythonAPI.zip \
    && unzip PythonAPI.zip \
    && rm PythonAPI.zip \
    && mkdir -p /opt/carla \
    && mv PythonAPI /opt/carla/PythonAPI

# Workaround/fix for using dpkg for cuda installation
# Only required for the lab PCs
RUN rm -f /var/lib/dpkg/info/fprintd.postinst \
    && rm -f /var/lib/dpkg/info/libfprint*.postinst \
    && rm -f /var/lib/dpkg/info/libpam-fprintd*.postinst \
    && dpkg --configure -a

# CUDA installation
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin --quiet \
    && mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb --quiet \
    && dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb \
    && cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install cuda \
    && rm cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb \
    && apt-get -y remove cuda-repo-ubuntu2004-11-8-local \
    && rm /etc/apt/preferences.d/cuda-repository-pin-600 \
    && rm -rf /var/cuda-repo-ubuntu1804-11-8-local/ \
    && rm /etc/apt/sources.list.d/cuda-ubuntu2004-11-8-local.list \
    && apt-get clean

ENV CUDA_HOME=/usr/local/cuda-11.8

# override python path, carla pip package path didn't exist and was using Python 3.7 instead of 2.7
ENV PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages
ENV PYTHONPATH=$PYTHONPATH:/opt/carla/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg:/opt/carla/PythonAPI/carla

# install mlocate, pip, wget, git and some ROS dependencies for building the CARLA ROS bridge
RUN apt-get update && apt-get install -y \
    mlocate python3-pip wget git python-is-python3 \
    ros-noetic-ackermann-msgs ros-noetic-derived-object-msgs \
    ros-noetic-carla-msgs ros-noetic-pcl-conversions \
    ros-noetic-rviz ros-noetic-rqt ros-noetic-pcl-ros ros-noetic-rosbridge-suite ros-noetic-rosbridge-server \
    ros-noetic-robot-pose-ekf ros-noetic-ros-numpy \
    ros-noetic-py-trees-ros ros-noetic-rqt-py-trees ros-noetic-rqt-reconfigure \
    ros-noetic-ros-pytest \
    ros-noetic-robot-localization

SHELL ["/bin/bash", "-c"]

# Create and apply the non-root user
# -f parameter to work in macos (https://unix.stackexchange.com/a/465014), since gid is already in use there
RUN groupadd --gid $USER_GID $USERNAME -f \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME

# create a catkin workspace
WORKDIR /catkin_ws
# make sure the catkin workspace belongs to the user
RUN sudo chown $USERNAME:$USERNAME /catkin_ws && mkdir src

# install the CARLA ROS bridge
ARG ROS_BRIDGE_GITHUB_URL=https://github.com/carla-simulator/ros-bridge
RUN git clone -b leaderboard-2.0 --recurse-submodules --single-branch $ROS_BRIDGE_GITHUB_URL src/ros-bridge
# apply patch to fix the build of carla_ros_bridge
RUN cd src/ros-bridge/carla_ackermann_control && \
    sed -i '24 i \  <build_depend condition="$ROS_VERSION == 1">carla_msgs</build_depend>' package.xml
RUN source /opt/ros/noetic/setup.bash && \
    rosdep update && rosdep install --from-paths src --ignore-src -r && \
    catkin_make install

ENV CARLA_ROS_BRIDGE_ROOT=/catkin_ws/src/ros-bridge

# avoid python warnings about missing bin directory in PATH
# (as we're not running as root, pip installs into ~/.local/bin)
ENV PATH=$PATH:/home/$USERNAME/.local/bin

# install pip packages
RUN python -m pip install pip --upgrade \
    && python -m pip install simple_pid pygame transformations roslibpy lxml black

# install the scenario runner from GitHub leaderboard-2.0 branch
ENV CARLA_ROOT=/opt/carla
ENV SCENARIO_RUNNER_ROOT=/opt/scenario_runner
RUN sudo mkdir $SCENARIO_RUNNER_ROOT && sudo chown $USERNAME:$USERNAME $SCENARIO_RUNNER_ROOT
RUN git clone -b leaderboard-2.0 --single-branch https://github.com/carla-simulator/scenario_runner.git $SCENARIO_RUNNER_ROOT
RUN echo 'pexpect' >> $SCENARIO_RUNNER_ROOT/requirements.txt && \
    python -m pip install -r $SCENARIO_RUNNER_ROOT/requirements.txt

# install the leaderboard from GitHub leaderboard-2.0 branch
ENV LEADERBOARD_ROOT=/opt/leaderboard
RUN sudo mkdir $LEADERBOARD_ROOT && sudo chown $USERNAME:$USERNAME $LEADERBOARD_ROOT
RUN git clone -b leaderboard-2.0 --single-branch https://github.com/una-auxme/leaderboard $LEADERBOARD_ROOT
RUN python -m pip install -r $LEADERBOARD_ROOT/requirements.txt
RUN sudo mkdir /opt/leaderboard-py3 && sudo chown $USERNAME:$USERNAME /opt/leaderboard-py3 && \
    ln -s $LEADERBOARD_ROOT/leaderboard /opt/leaderboard-py3/leaderboard && \
    ln -s $SCENARIO_RUNNER_ROOT/srunner /opt/leaderboard-py3/srunner

# environment variables for the leaderboard_evaluator
ENV ROUTES=$LEADERBOARD_ROOT/data/routes_devtest.xml
ENV REPETITIONS=1
ENV DEBUG_CHALLENGE=1
#ENV TEAM_AGENT=$LEADERBOARD_ROOT/leaderboard/autoagents/human_agent.py
ENV TEAM_AGENT=/workspace/code/agent/src/agent/agent.py
ENV CHECKPOINT_ENDPOINT=/workspace/code/simulation_results.json
ENV CHALLENGE_TRACK=MAP

ENV CARLA_SIM_HOST=localhost
ENV CARLA_SIM_WAIT_SECS=15
ENV SCENARIO_RUNNER_PATH=/opt/scenario_runner

# setup python path
RUN echo /catkin_ws/install/lib/python3/dist-packages >> /home/$USERNAME/.local/lib/python3.8/site-packages/carla.pth && \
    echo /catkin_ws/devel/lib/python3/dist-packages >> /home/$USERNAME/.local/lib/python3.8/site-packages/carla.pth && \
    echo /opt/ros/noetic/lib/python3/dist-packages >> /home/$USERNAME/.local/lib/python3.8/site-packages/carla.pth && \
    echo /opt/ros/noetic/lib/python3/dist-packages >> /home/$USERNAME/.local/lib/python3.8/site-packages/carla.pth && \
    echo /opt/leaderboard-py3 >> /home/$USERNAME/.local/lib/python3.8/site-packages/carla.pth && \
    echo /opt/carla/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg >> /home/$USERNAME/.local/lib/python3.8/site-packages/carla.pth && \
    echo /opt/carla/PythonAPI/carla >> /home/$USERNAME/.local/lib/python3.8/site-packages/carla.pth

RUN sudo mkdir /workspace && sudo chown $USERNAME:$USERNAME /workspace

RUN echo "export CUDA_HOME=/usr/local/cuda-11" >> ~/.bashrc \
    && echo "export PATH=$PATH:$CUDA_HOME/bin" >> ~/.bashrc

COPY --chown=$USERNAME:$USERNAME ./code/requirements.txt /workspace/
ENV TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6 8.7 8.9+PTX"
ENV IABN_FORCE_CUDA=1

RUN source ~/.bashrc && pip install -r /workspace/requirements.txt

# Add agent code
COPY --chown=$USERNAME:$USERNAME ./code /workspace/code/

# For debugger
ENV PAF_CATKIN_CODE_ROOT=/catkin_ws/src/code
ENV PAF_WORKSPACE_CODE_ROOT=/workspace/code
# Link code into catkin workspace
RUN ln -s -T ${PAF_WORKSPACE_CODE_ROOT} ${PAF_CATKIN_CODE_ROOT}

# re-make the catkin workspace
RUN source /opt/ros/noetic/setup.bash && catkin_make

ADD ./build/docker/agent/entrypoint.sh /entrypoint.sh

# set the default working directory to the code
WORKDIR ${PAF_WORKSPACE_CODE_ROOT}

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash", "-c", "sleep 10 && \
    python3 /opt/leaderboard/leaderboard/leaderboard_evaluator.py \
    --debug=${DEBUG_CHALLENGE} \
    --repetitions=${REPETITIONS} \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --track=${CHALLENGE_TRACK} \
    --agent=${TEAM_AGENT} \
    --routes=${ROUTES} \
    --host=${CARLA_SIM_HOST}"]

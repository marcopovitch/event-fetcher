#!/usr/bin/env python
import os
import _pickle as cPickle
import logging
import os.path
import re
import sys
import warnings
import numpy as np
import argparse
import yaml
import urllib.parse
import urllib.error
import pandas as pd
from icecream import ic

from obspy import Inventory
from obspy import Stream, read_events, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.filesystem.sds import Client as ClientSDS
from obspy.geodetics import gps2dist_azimuth

# Denoiser
from keras.models import load_model
import seisbench.models as sbm

# Denoiser: https://github.com/JanisHe/seisDAE
SEIS_DAE = "/Users/marc/github/seisDAE"
sys.path.append(SEIS_DAE)
from denoiser.denoise_utils import denoising_stream


# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("EventFetcher")
logger.setLevel(logging.INFO)

MODEL = os.path.join(SEIS_DAE, "Models", "gr_mixed_stft.h5")
CONFIG = os.path.join(SEIS_DAE, "config", "gr_mixed_stft.config")


def phasenet_dump(traces, directory):
    logger.info("PhaseNet dump:")
    try:
        os.makedirs(directory, exist_ok=True)
        logger.debug("Directory '%s' created successfully." % directory)
    except OSError as error:
        logger.error("Directory '%s' can not be created !" % directory)

    # get net.cha.loc.* from all traces
    wfids = set()
    for tr in traces:
        wfids.add(".".join(tr.id.split(".")[:3]))
    wfids = list(map(lambda x: x + ".*", wfids))

    for wfid in wfids:
        net_sta_loc = ".".join(wfid.split(".")[:3])
        st = traces.select(id=wfid)
        filename = os.path.join(
            directory,
            f"{net_sta_loc}.mseed",
        )
        st.write(filename, format="MSEED")

    # generates chan.txt for dbclust
    chantxt_filename = os.path.join(directory, "chan.txt")
    logger.debug(f"Generating {chantxt_filename}")
    with open(chantxt_filename, "w") as fp:
        for wfid in wfids:
            st = traces.select(id=wfid).sort(["channel"], reverse=True)
            for tr in st:
                s = tr.stats
                fp.write(f"{s.network}_{s.station}_{s.location}_{s.channel}\n")

    # generates csv file
    csv_filename = os.path.join(directory, "fname.csv")
    logger.debug(f"Generating {csv_filename}")
    with open(csv_filename, "w") as fp:
        fp.write("fname,E,N,Z\n")
        for wfid in wfids:
            filename = ".".join(wfid.split(".")[:3]) + ".mseed"
            st = traces.select(id=wfid)
            Z_trace = st.select(component="Z")[0]
            st.remove(Z_trace)
            st.sort(["channel"], reverse=False)
            try:
                fp.write(
                    f"{filename},{st[0].stats.channel},{st[1].stats.channel},{Z_trace.stats.channel}\n"
                )
            except Exception as e:
                logger.error(e)
                logger.error(f"Something went wrong getting 3 components in {wfid}:")
                logger.error(st)


def mseed_dump_by_trace(traces, directory, station_json=False):
    """
    Dump traces to MiniSEED files and optionally generate a JSON station file.

    Args:
        traces (list): List of traces to be dumped.
        directory (str): Directory path where the MiniSEED files will be saved.
        station_json (bool, optional): Whether to generate a JSON station file. Defaults to False.
    """
    logger.info("Mseed dump:")
    try:
        os.makedirs(directory, exist_ok=True)
        logger.debug("Directory '%s' created successfully." % directory)
    except OSError as error:
        logger.error("Directory '%s' can not be created !" % directory)

    for tr in traces:
        stats = tr.stats
        filename = os.path.join(
            directory,
            f"{stats.network}.{stats.station}.{stats.location}.{stats.channel}.{stats.starttime}.{stats.endtime}",
        )
        tr.write(filename + ".mseed", format="MSEED")
        stats.response.write(filename + ".xml", format="STATIONXML")

    # generates json station file
    if station_json:
        pass


def mseed_dump_by_station(traces, directory):
    """
    Dump station traces to MiniSEED files and corresponding StationXML files.

    Args:
        traces (obspy.core.stream.Stream): The traces to be dumped.
        directory (str): The directory where the files will be saved.

    Returns:
        None
    """
    logger.info("Mseed dump:")
    try:
        os.makedirs(directory, exist_ok=True)
        logger.debug("Directory '%s' created successfully." % directory)
    except OSError as error:
        logger.error("Directory '%s' can not be created !" % directory)

    id_list = []
    for tr in traces:
        id_list.append(tr.id.split("."))

    df = pd.DataFrame(id_list, columns=["net", "sta", "loc", "chan"])
    for g in df.groupby(["net", "sta", "loc"]):
        st = traces.select(network=g[0][0], station=g[0][1], location=g[0][2])
        stats = st[0].stats
        filename = os.path.join(
            directory,
            f"{stats.network}.{stats.station}.{stats.location}.{stats.starttime}.{stats.endtime}",
        )
        st.write(filename + ".mseed", format="MSEED")
        stats.response.write(filename + ".xml", format="STATIONXML")


def inventory2df(inventory: Inventory) -> pd.DataFrame:
    """Convert inventory to dataframe

    Args:
        inventory (Inventory): inventory to convert

    Returns:
        pd.DataFrame: dataframe
    """
    channels_info = []
    for network in inventory:
        for station in network:
            for channel in station.channels:

                if channel.response and channel.response.instrument_sensitivity:
                    scale = channel.response.instrument_sensitivity.value
                    scale_freq = channel.response.instrument_sensitivity.frequency
                    scale_units = channel.response.instrument_sensitivity.input_units
                else:
                    scale = scale_freq = scale_units = None

                if channel.sensor:
                    sensor_description = channel.sensor.description
                else:
                    sensor_description = None

                channel_info = {
                    "Network": network.code,
                    "Station": station.code,
                    "Location": channel.location_code,
                    "Channel": channel.code,
                    "Latitude": station.latitude,
                    "Longitude": station.longitude,
                    "Elevation": station.elevation,
                    "Depth": channel.depth,
                    # "Azimuth": channel.azimuth,
                    # "Dip": channel.dip,
                    # "SensorDescription": sensor_description,
                    # "Scale": scale,
                    # "ScaleFreq": scale_freq,
                    # "ScaleUnits": scale_units,
                    "SampleRate": channel.sample_rate,
                    "StartTime": station.start_date,
                    "EndTime": station.end_date,
                }
                # Add channel information to the list
                channels_info.append(channel_info)

    # Create a pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(channels_info, dtype=str)
    if df.empty:
        return df

    df["SampleRate"] = df["SampleRate"].apply(np.float32)
    df = df.fillna("")
    df["Latitude"] = df["Latitude"].apply(np.float32)
    df["Longitude"] = df["Longitude"].apply(np.float32)
    df["Elevation"] = df["Elevation"].apply(np.float32)
    df["Location"] = df["Location"].astype(str)
    df["Channel"] = df["Channel"].astype(str)

    df = df.drop_duplicates()

    return df


def filter_out_station_without_3channels(waveforms_id, bulk, inventory, txt):
    tmp_bulk = []
    for net, sta, loc, chan, t1, t2 in bulk:
        rqt_time = t1 + (t2 - t1) / 2.0
        inv = inventory.select(network=net, station=sta, location=loc, time=rqt_time)

        # limits only to channels with the highest sample rate
        df = inventory2df(inv)
        if df.empty:
            logger.error(f"No metadata for {net}.{sta}.{loc}.{chan}")
            continue

        max_sample_rate = df["SampleRate"].max()
        df = df[df["SampleRate"] == max_sample_rate]
        df = df.sort_values(by="StartTime")[:3]
        chan = df["Channel"].tolist()[0]
        chan = chan[:2] + "?"

        if len(df) == 3:
            tmp_bulk.append((net, sta, loc, chan, t1, t2))
        else:
            w = ".".join((net, sta, loc, chan))
            if inv:
                logger.debug(
                    "[%s] Filtering out %s (only %d channel(s))"
                    % (txt, w, len(inv[0][0]))
                )
            else:
                logger.warning(
                    "[%s] Filtering out %s (no metadata at %s)" % (txt, w, rqt_time)
                )
            id = ".".join((net, sta, loc, chan))
            waveforms_id = cleanup_waveforms_id(waveforms_id, id)
    return waveforms_id, tmp_bulk


def filter_out_station_by_distance(
    waveforms_id, bulk, inventory, event, station_max_dist_km
):
    tmp_bulk = []
    for net, sta, loc, chan, t1, t2 in bulk:
        tmpchan = chan[:-1] + "Z"
        w = ".".join((net, sta, loc, tmpchan))
        t = t1 + (t2 - t1) / 2.0
        try:
            coord = inventory.get_coordinates(w, t)
        except Exception as e:
            logger.error("[%s] %s (%s, %s): %s", event.id, w, t1, t2, e)
            continue

        distance, az, baz = gps2dist_azimuth(
            coord["latitude"],
            coord["longitude"],
            event.latitude,
            event.longitude,
        )
        # distance in meters, convert it to km
        distance = distance / 1000.0
        if distance <= station_max_dist_km:
            tmp_bulk.append((net, sta, loc, chan, t1, t2))
        else:
            logger.debug(
                "Filtering out %s (dist(%.1f) > %.1f)"
                % (w, distance, station_max_dist_km)
            )
            id = ".".join((net, sta, loc, chan))
            waveforms_id = cleanup_waveforms_id(waveforms_id, id)
    return waveforms_id, tmp_bulk


def cleanup_waveforms_id(waveforms_id, id):
    net, sta, loc, chan = id.split(".")
    wid_to_remove = []
    for wid in waveforms_id:
        wid_net, wid_sta, wid_loc, wid_chan = wid.split(".")
        if wid_net == net and wid_sta == sta:
            wid_to_remove.append(wid)

    for wid in wid_to_remove:
        waveforms_id.remove(wid)

    return waveforms_id


def remove_flat_traces(waveforms_id, traces, txt):
    # variance is used to detect flat signal
    tolerance = 1e-5
    traces_to_remove = []
    for i, trace in enumerate(traces):
        variance = np.var(trace.data)
        if variance < tolerance:
            traces_to_remove.append(trace)

    for tr in traces_to_remove:
        net_sta_loc = ".".join(tr.id.split(".")[:3])
        logger.warning(
            "[%s] Flat channel for %s detected: removing trace %s"
            % (txt, net_sta_loc, tr.id)
        )
        traces.remove(tr)
        cleanup_waveforms_id(waveforms_id, tr.id)

    return waveforms_id


def remove_traces_without_3channels(waveforms_id, traces, txt):
    traces_done = []
    traces_to_remove = []
    for i, trace in enumerate(traces):
        stats = trace.stats
        net_sta_loc = ".".join([stats.network, stats.station, stats.location])
        if net_sta_loc in traces_done:
            continue

        tmp = traces.select(
            network=stats.network, station=stats.station, location=stats.location
        )
        if tmp.count() != 3:
            for tr in tmp:
                traces_to_remove.append(tr)
        traces_done.append(net_sta_loc)

    for tr in traces_to_remove:
        net_sta_loc = ".".join(tr.id.split(".")[:3])
        logger.warning(
            "[%s] Missing channel for %s: removing trace %s" % (txt, net_sta_loc, tr.id)
        )
        traces.remove(tr)
        cleanup_waveforms_id(waveforms_id, tr.id)

    return waveforms_id


class EventInfo(object):
    """Store basic event information
    - latitude
    - longitude
    - T0
    - magnitude
    - magnitude_type
    - event_type
    - qml
    """

    def __init__(self):
        self.id = None
        self.latitude = None
        self.longitude = None
        self.depth = None
        self.T0 = None
        self.magnitude = None
        self.magnitude_type = None
        self.event_type = None
        self.qml = None

    def __str__(self):
        try:
            mystr = (
                f"event_id={self.id}, {self.event_type}\n"
                f"T0={self.T0}, lat={self.latitude:.5f}, lon={self.longitude:.5f}, depth_km={self.depth:.1f}\n"
                f"magnitude={self.magnitude:.2f} {self.magnitude_type}"
            )
        except Exception as e:
            logger.error(self.__dict__)
            mystr = ""
        return mystr


class EventFetcher(object):
    """Fetch qml and traces for a given event id."""

    def __init__(
        self,
        event_id,
        starttime=None,
        endtime=None,
        starttime_offset=0,
        time_length=60,
        station_max_dist_km=None,
        base_url=None,
        ws_event_url=None,
        ws_station_url=None,
        ws_dataselect_url=None,
        sds=None,
        black_listed_waveforms_id=None,
        waveforms_id=None,
        use_only_trace_with_weighted_arrival=True,
        keep_only_3channels_station=False,
        enable_RTrotation=False,
        enable_denoising=False,
        denoise_model="original",
        backup_dirname=".",
        enable_read_cache=False,
        enable_write_cache=False,
        write_cache_format="pickle",
        fdsn_debug=False,
        log_level=logging.INFO,
    ):
        logger.setLevel(log_level)
        self.st = None
        self.starttime = starttime
        self.endtime = endtime
        self.starttime_offset = starttime_offset
        self.time_length = time_length
        self.station_max_dist_km = station_max_dist_km
        self.use_only_trace_with_weighted_arrival = use_only_trace_with_weighted_arrival
        self.keep_only_3channels_station = keep_only_3channels_station
        self.enable_RTrotation = enable_RTrotation
        self.enable_denoising = enable_denoising
        self.denoise_model = denoise_model
        # cache
        self.enable_read_cache = enable_read_cache
        self.enable_write_cache = enable_write_cache
        self.write_cache_format = write_cache_format
        # fdsn or sds
        self.sds = sds
        self.fdsn_debug = fdsn_debug
        self.base_url = base_url
        self.ws_event_url = ws_event_url
        self.ws_station_url = ws_station_url
        self.ws_dataselect_url = ws_dataselect_url
        self.trace_client = None

        if not os.path.isdir(backup_dirname):
            try:
                os.makedirs(backup_dirname, exist_ok=True)
                logger.debug("set up %s as cache directory", backup_dirname)
            except Exception as e:
                logger.error(
                    "Can't create cache directory '%s' (%s) !", backup_dirname, e
                )
                self.event = EventInfo()
                return

        self.backup_event_file = os.path.join(
            backup_dirname, f"{urllib.parse.quote(event_id, safe='')}.qml"
        )
        self.backup_traces_file = os.path.join(backup_dirname, "waveforms")

        if black_listed_waveforms_id:
            self.black_listed_waveforms_id = black_listed_waveforms_id
        else:
            self.black_listed_waveforms_id = []

        self.event = EventInfo()
        self.event.id = event_id
        self._fetch_data(waveforms_id=waveforms_id)
        # try:
        #     self._fetch_data(waveforms_id=waveforms_id)
        # except Exception as e:
        #     logger.error(f"Can't get traces for event_id {event_id}")
        #     logger.error(f"error: {e}")
        #     self.st = []
        #     return

        self.get_picks()
        if self.st == []:
            return
        elif self.st is None:
            self.st = []
            return

        self.compute_distance_az_baz()

        if self.enable_RTrotation and self.st:
            st_RT = self.rotate_to_RT()
            self.st += st_RT

        # if a component is shorter, force same signal length (e.g. after rotation)
        try:
            self.st._trim_common_channels()
        except Exception as e:
            logger.warning("(%s) can't _trim_common_channels(): %s", self.event.id, e)

        # Sync all traces to starttime and endtime ... but could produce masked array
        self.st.trim(starttime=starttime, endtime=endtime)

        # save traces with pickle
        if self.enable_write_cache and self.backup_traces_file:
            logger.debug("writting to %s", self.backup_traces_file)
            if self.write_cache_format == "pickle":
                with open(self.backup_traces_file, "wb") as fp:
                    cPickle.dump(self.st, fp)
            elif self.write_cache_format == "mseed":
                try:
                    # mseed_dump_by_trace(self.st, self.backup_traces_file)
                    mseed_dump_by_station(self.st, self.backup_traces_file)
                except Exception as e:
                    logger.error(e)
                    return
            elif self.write_cache_format == "phasenet":
                try:
                    phasenet_dump(self.st, self.backup_traces_file)
                except Exception as e:
                    logger.error(e)
                    return

        if self.st:
            self.st.sort()
            if logger.level == logging.DEBUG:
                logger.debug(self.st.__str__(extended=True))
            # else:
            #    logger.info("%s %s", self.event.id, self.st)
        else:
            logger.warning("No trace (%s) in _fetch_data() !", self.event.id)

    def _fetch_data(self, waveforms_id=None):
        # Fetch event's traces from ws or cached files
        cat = None
        fetch_from_cache_success = None

        if self.enable_read_cache:
            if os.path.isfile(self.backup_event_file):
                logger.debug(
                    "Fetching event %s from file %s.",
                    self.event.id,
                    self.backup_event_file,
                )
                cat = read_events(self.backup_event_file)
                fetch_from_cache_success = True
            else:
                logger.debug(
                    "Trying to fetch event %s from file. But %s does not exist!",
                    self.event.id,
                    self.backup_event_file,
                )
                fetch_from_cache_success = False

        if not self.enable_read_cache or not fetch_from_cache_success:
            self.event_client = Client(
                debug=self.fdsn_debug, service_mappings={"event": self.ws_event_url, "dataselect": None, "station": None}
            )

            logger.debug("Fetching event %s from FDSN-WS.", self.event.id)
            cat = self.get_event()

        if not cat:
            # logger.error("(%s) No event found !" % self.event.id)
            return

        try:
            self.event.qml = cat.events[0]
        except Exception as e:
            logger.error(self.event.id, e)
            return

        (
            self.event.latitude,
            self.event.longitude,
            self.event.depth,
        ) = self.get_event_coordinates(self.event.qml)
        self.event.T0 = self.get_event_time(self.event.qml)
        self.event.event_type = self.get_event_type(self.event.qml)
        self.event.magnitude = self.get_magnitude(self.event.qml)
        self.event.magnitude_type = self.get_magnitude_type(self.event.qml)
        logger.debug(self.event)

        if waveforms_id:
            self.waveforms_id = waveforms_id
        else:
            logger.debug(
                "Use only traces with weight > 0 : %s",
                self.use_only_trace_with_weighted_arrival,
            )

            if self.use_only_trace_with_weighted_arrival:
                logger.debug("Using: get_event_waveforms_id")
                self.waveforms_id = self._hack_streams(
                    self.get_event_waveforms_id(self.event.qml)
                )
                self.show_pick_offet(self.event.qml)
            else:
                logger.debug("Using: get_event_waveforms_id_within_distance")
                self.waveforms_id = self._hack_streams(
                    self.get_event_waveforms_id_within_distance(
                        self.event.qml, self.station_max_dist_km
                    )
                )

        # Set time window for trace extraction
        self._set_extraction_time_window()

        fetch_from_cache_success = None
        if self.enable_read_cache:
            if os.path.isfile(self.backup_traces_file):
                logger.debug(
                    "Fetching traces from cached file %s.", self.backup_traces_file
                )
                with open(self.backup_traces_file, "rb") as fp:
                    self.st = cPickle.load(
                        fp, fix_imports=True, encoding="ASCII", errors="strict"
                    )
                fetch_from_cache_success = True
            else:
                logger.debug(
                    "Trying to fetch traces from cached file, but %s does not exist!",
                    self.backup_event_file,
                )
                fetch_from_cache_success = False

        if not self.enable_read_cache or not fetch_from_cache_success:
            # set FDSN clients
            # configuring 3 differents urls doesn't work.
            # we have to split in 2 Fdsn clients trace and event
            if not self.trace_client:
                self.trace_client = Client(
                    debug=self.fdsn_debug,
                    base_url=self.base_url,
                    service_mappings={
                        "event": None,
                        "dataselect": self.ws_dataselect_url,
                        "station": self.ws_station_url,
                    },
                    timeout=300,
                )

            # Use SDS (seiscomp data struture) to get traces rather than fdsn-dataselect
            if self.sds:
                self.trace_client_sds = ClientSDS(self.sds)

            logger.debug("Fetching traces (%s) from FDSN-WS or SDS", self.event.id)
            # self.st = self.get_trace(self.starttime, self.endtime)
            try:
                self.st = self.get_trace_bulk(self.starttime, self.endtime)
            except urllib.error.HTTPError as e:
                raise e

        if self.enable_denoising:
            # denoise_model can be 'dae', 'original' or 'urban'.
            logger.info(f"Denoising traces ... with model {self}")
            self.st = denoise_stream(self.st, model_name=self.denoise_model)
            print(self.st)

        # remove black listed channels
        # to be optimized (at inventory level if possible)
        # self._remove_from_stream(self.black_listed_waveforms_id)

        if self.st == []:
            logger.warning("No traces (%s)!" % self.event.id)

    def _set_extraction_time_window(self):
        """Set time window for trace extraction"""
        if self.starttime is None:
            self.starttime = self.event.T0
        self.starttime += self.starttime_offset

        if self.endtime is None:
            self.endtime = self.starttime + self.time_length

    def _hack_P_stream(self, waveforms_id):
        """Hack to get rid off sc3 users mislabeling phases."""
        net, sta, loc, chan = waveforms_id.split(".")
        if len(chan) == 3:
            if chan[-1] == "-" or chan[-1] == "?":
                chan = chan[:2] + "Z"
                return ".".join([net, sta, loc, chan])

        if len(chan) == 2:
            # eg: RT.MTT2..BH
            chan = chan + "Z"
            return ".".join([net, sta, loc, chan])

        return waveforms_id

    def _hack_streams(self, waveforms_ids):
        """Hack to get rid off sc3 users mislabeling phases."""
        wfid_list = set()
        for w in waveforms_ids:
            w_fixed = self._hack_P_stream(w)
            net, sta, loc, chan = w_fixed.split(".")
            chan = chan[:2] + "?"
            wfid_list.add(".".join([net, sta, loc, chan]))
        return list(wfid_list)

    def _remove_from_stream(self, waveforms_id_list):
        # remove black listed waveform_id
        # should be optimized (to be done at the inventory level, if possible)
        # for net, sta, loc, chan in waveforms_id_list:
        #   wfid = f"{net}.{sta}.{loc}.{chan}"
        for wfid in waveforms_id_list:
            net, sta, loc, chan = wfid.split(".")
            for tr in self.st.select(
                network=net, station=sta, location=loc, channel=chan
            ):
                try:
                    self.st.remove(tr)
                except Exception as e:
                    logger.debug(f"Can't remove trace {wfid} ({e})")
                else:
                    logger.debug(f"Removed black listed trace fid {wfid}")

    def get_trace_bulk(self, starttime, endtime):
        logger.debug(f"{self.event.id}: building station list ...")
        bulk = []
        for w in self.waveforms_id:
            # check if black listed using regex
            if any(re.match(b, w) for b in self.black_listed_waveforms_id):
                logger.info(f"{self.event.id}: ignoring black listed {w} !")
                continue

            logger.debug(f"{self.event.id}: adding station {w}")
            net, sta, loc, chan = w.split(".")
            bulk.append((net, sta, loc, chan, starttime, endtime))

        # get inventory
        logger.debug(f"{self.event.id}: getting station inventory ...")
        try:
            inventory = self.trace_client.get_stations_bulk(bulk, level="response")
        except Exception as e:
            logger.error(f"{self.event.id}: {type(e).__name__} - {str(e)}")
            return Stream()

        # keep only stations with 3 component (using inventory info only)
        if self.keep_only_3channels_station:
            self.waveforms_id, bulk = filter_out_station_without_3channels(
                self.waveforms_id, bulk, inventory, self.event.id
            )

        # get rid off stations too far away
        if self.station_max_dist_km:
            self.waveforms_id, bulk = filter_out_station_by_distance(
                self.waveforms_id,
                bulk,
                inventory,
                self.event,
                self.station_max_dist_km,
            )

        # get traces but without response as attach_response does not work as expected
        logger.debug(f"{self.event.id}: getting waveforms ...")
        try:
            if self.sds:
                # Use SDS (seiscomp data struture) to get traces rather than fdsn-dataselect
                traces = self.trace_client_sds.get_waveforms_bulk(bulk)
            else:
                traces = self.trace_client.get_waveforms_bulk(
                    bulk, attach_response=False
                )
        except Exception as e:
            # logger.error("%s %s", e, self.event.id)
            # logger.error(e.status_code)
            raise e
            # return Stream()

        # merge multiple segments if any
        try:
            traces.merge(method=0, fill_value="interpolate")
        except Exception as e:
            logger.error("(merge) %s %s", e, self.event.id)
            return Stream()

        # add inventory to trace
        for i, _w in enumerate(traces):
            _stats = _w.stats
            _wid = ".".join(
                [_stats.network, _stats.station, _stats.location, _stats.channel]
            )
            logger.debug(_wid)
            traces[i].stats.response = inventory.select(
                network=_stats.network,
                station=_stats.station,
                location=_stats.location,
                # channel=_stats.channel,  # all channel have to be included for rotation !!??!!
                time=starttime + (endtime - starttime) / 2.0,
            )
            logger.debug(traces[i].stats.response)

            try:
                traces[i].stats.coordinates = traces[i].stats.response.get_coordinates(
                    _wid
                )
            except Exception as e:
                logger.error(
                    "(%s) No station coordinates for %s (%s)" % (self.event.id, _wid, e)
                )
                traces[i].stats.coordinates = None
            logger.debug("%s: %s", _wid, traces[i].stats.coordinates)

        # remove "flat" traces (with same value everywhere)
        self.waveforms_id = remove_flat_traces(self.waveforms_id, traces, self.event.id)

        # Check if 3 channels are present (ie. no missing trace)
        if self.keep_only_3channels_station:
            self.waveforms_id = remove_traces_without_3channels(
                self.waveforms_id, traces, self.event.id
            )

        # Sync all traces to starttime
        traces.trim(starttime=starttime, endtime=endtime)

        return traces

    def get_trace(self, starttime, endtime):
        """Get waveform using FDSNWS"""
        traces = Stream()
        # print(self.waveforms_id)
        for w in self.waveforms_id:
            logger.debug("Working on %s ... ", w)
            net, sta, loc, chan = w.split(".")

            # get trace
            logger.debug("Start to fetch trace %s [%s-%s]", w, starttime, endtime)
            try:
                waveform = self.trace_client.get_waveforms(
                    net, sta, loc, chan, starttime, endtime, attach_response=False
                )
            except Exception as e:
                logger.error("(get_trace/wf)%s %s", e, self.event.id)
                continue

            if not waveform:
                logger.debug("No data for trace %s [%s-%s]", w, starttime, endtime)
                continue

            # be sure to have only one segment in trace
            try:
                waveform.merge(method=0, fill_value="interpolate")
            except Exception as e:
                logger.warning("%s %s", self.event.id, e)
                logger.warning(waveform)
                continue
            else:
                logger.debug(waveform)

            # get coordinates since attach_response seems not to be enough
            logger.debug("Start to fetch inventory for %s", w)
            try:
                inventory = self.trace_client.get_stations(
                    network=net,
                    station=sta,
                    location=loc,
                    channel=chan,
                    starttime=starttime,
                    endtime=endtime,
                    level="response",
                )
            except Exception as e:
                logger.error("(get_trace/inv)%s %s", e, self.event.id)
                continue

            logger.debug(inventory)

            for i, _w in enumerate(waveform):
                _stats = _w.stats
                _wid = ".".join(
                    [_stats.network, _stats.station, _stats.location, _stats.channel]
                )
                logger.debug(_wid)
                waveform[i].stats.response = inventory
                try:
                    waveform[i].stats.coordinates = inventory.get_coordinates(_wid)
                except Exception as e:
                    logger.error("%s %s", e, self.event.id)
                    waveform[i].stats.coordinates = None
                logger.debug("%s: %s", _wid, waveform[i].stats.coordinates)

            # store trace
            traces += waveform

        # Sync all traces to starttime
        traces.trim(starttime=starttime, endtime=endtime)

        return traces

    def rotate_to_RT(self):
        # make a copy and rotate traces
        # return only R and T traces
        if not hasattr(self, "waveforms_id"):
            return

        wids = []
        for w in self.waveforms_id:
            logger.debug("Working on %s ... ", w)
            net, sta, loc, chan = w.split(".")
            wids.append(".".join((net, sta, loc, "*")))
        wids = set(wids)

        # print(wids)
        st_RT = Stream()
        stcopy = self.st.copy()

        for wid in wids:
            st = stcopy.select(id=wid)
            try:
                st._trim_common_channels()
            except Exception as e:
                logger.warning(
                    "(%s) in rotate_to_RT(), can't trim: %s (%s)", self.event.id, wid, e
                )

            try:
                logger.debug("Rotating %s" % wid)
                inventory = st[0].stats.response  # All channel should be included here
                # nb_channel = len(inventory.get_contents()["channels"])
                # if nb_channel != 3:
                #    logger.warning("%s has only %d channel" % (wid, nb_channel))
                #    raise
                st.rotate(method="->ZNE", inventory=inventory)
                st.rotate(method="NE->RT", inventory=inventory)
            except IndexError:
                logger.warning("(%s) Can't rotate: %s (no data !)", self.event.id, wid)
            except Exception as e:
                logger.warning("(%s) Can't rotate: %s (%s)", self.event.id, wid, e)
            else:
                # logger.warning(st)
                # remove Z trace
                for tr in st.select(component="Z"):
                    st.remove(tr)
                st_RT += st

        return st_RT

    def get_event(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            try:
                cat = self.event_client.get_events(
                    eventid=self.event.id, includearrivals=True
                )
            except Exception as e:
                logger.error("Error getting event %s" % self.event.id)
                logger.debug(e)
                return None

            if len(cat.events) == 0:
                logger.error("Empty event %s !" % self.event.id)
                return None

            if self.enable_write_cache and self.backup_event_file:
                logger.debug(
                    "writting event (%s) to quakeml file %s",
                    self.event.id,
                    self.backup_event_file,
                )
                cat.write(self.backup_event_file, format="QUAKEML")
        return cat

    def get_event_coordinates(self, e):
        o = e.preferred_origin()
        return o.latitude, o.longitude, o.depth / 1000.0

    def get_event_time(self, e):
        o = e.preferred_origin()
        return o.time

    def get_event_type(self, e):
        return e.event_type

    def get_magnitude(self, e):
        if len(e.magnitudes):
            return e.magnitudes[0].mag
        else:
            return None

    def get_magnitude_type(self, e):
        if len(e.magnitudes):
            return e.magnitudes[0].magnitude_type
        else:
            return None

    def get_event_waveforms_id(self, e):
        waveforms_id = []
        o = e.preferred_origin()
        for a in o.arrivals:
            if a.time_weight == 0.0 and self.use_only_trace_with_weighted_arrival:
                continue
            for p in e.picks:
                if a.pick_id == p.resource_id:
                    wfid = self._hack_P_stream(p.waveform_id.get_seed_string())
                    waveforms_id.append(wfid)
                    logger.debug("Adding %s", wfid)
                    # waveforms_id.append(p.waveform_id.get_seed_string())
                    break
        return waveforms_id

    def get_event_waveforms_id_within_distance(self, e, dist_km):
        if dist_km is None:
            logger.error(
                "When using use_only_trace_with_weighted_arrival=False,  station_max_dist_km must be defined !"
            )
            sys.exit(1)

        o = e.preferred_origin()
        t0 = o.time
        # get waveform_id of all stations within dist_km radius
        logger.debug(
            "Start to fetch waveform_id for %s with %d km radius",
            self.event.id,
            dist_km,
        )

        if not self.trace_client:
            self.trace_client = Client(
                debug=self.fdsn_debug,
                base_url=self.base_url,
                service_mappings={
                    "event": None,
                    "dataselect": self.ws_dataselect_url,
                    "station": self.ws_station_url,
                },
                timeout=300,
            )

        try:
            inventory = self.trace_client.get_stations(
                starttime=t0,
                endtime=t0,
                level="channel",
                latitude=self.event.latitude,
                longitude=self.event.longitude,
                minradius=0,
                maxradius=dist_km / 111.0,  # dist in degres
                includerestricted=True,
            )
        except Exception as e:
            logger.error(
                "(get_event_waveforms_id_within_distance) %s %s", e, self.event.id
            )
            return []

        waveforms_id = []
        for net in inventory:
            for sta in net:
                # FIXME: get data with the higher sampling rate only
                for chan in sta.select(channel="[SBHED][HNP]Z"):
                    wf_id = ".".join(
                        [net.code, sta.code, chan.location_code, chan.code]
                    )
                    waveforms_id.append(wf_id)
                    # logger.debug(f"{wf_id} is in range.")
        return waveforms_id

    def compute_distance_az_baz(self):
        # Calculating distance and azimuth from station to event
        if not self.st:
            return
        for tr in self.st:
            if "coordinates" not in tr.stats or tr.stats.coordinates is None:
                logger.warning(
                    "(%s) compute_distance_az_baz: no coordinates for %s"
                    % (self.event.id, tr)
                )
                continue

            distance, az, baz = gps2dist_azimuth(
                tr.stats.coordinates.latitude,
                tr.stats.coordinates.longitude,
                self.event.latitude,
                self.event.longitude,
            )
            tr.stats.distance = distance  # in meters
            tr.stats.back_azimuth = az

    def get_picks(self, e=None):
        self.picks = {}
        if e is None:
            e = self.event.qml
            if e is None:
                return

        o = e.preferred_origin()
        t0 = o.time
        for a in o.arrivals:
            if not a.phase.startswith("P"):
                logger.debug("Looking for P phase: ignoring %s !", a.phase)
                continue
            for p in e.picks:
                if a.pick_id == p.resource_id:
                    wfid = self._hack_P_stream(p.waveform_id.get_seed_string())
                    self.picks[wfid] = {
                        "time": p.time,
                        "offset": p.time - (t0 + self.starttime_offset),
                    }
                    break

    def show_pick_offet(self, e=None):
        if e is None:
            e = self.event.qml

        o = e.preferred_origin()
        t0 = o.time
        for a in o.arrivals:
            if not a.phase.startswith("P"):
                continue
            for p in e.picks:
                if a.pick_id == p.resource_id:
                    logger.debug(
                        "%s %s %s",
                        self._hack_P_stream(p.waveform_id.get_seed_string()),
                        p.time,
                        p.time - t0,
                    )
                    break


def _test(event_id):
    # webservice URL
    ws_base_url = "http://10.0.1.36"
    ws_event_url = "http://10.0.1.36:8080/fdsnws/event/1"
    ws_station_url = "http://10.0.1.36:8080/fdsnws/station/1"
    ws_dataselect_url = "http://10.0.1.36:8080/fdsnws/dataselect/1"

    # get data
    mydata = EventFetcher(
        event_id,
        time_length=90,
        starttime_offset=-10,
        station_max_dist_km=200,
        base_url=ws_base_url,
        ws_event_url=ws_event_url,
        ws_station_url=ws_station_url,
        ws_dataselect_url=ws_dataselect_url,
        use_only_trace_with_weighted_arrival=False,
        keep_only_3channels_station=True,
        enable_denoising=False,
        enable_RTrotation=False,
        backup_dirname=event_id,
        enable_write_cache=True,
        enable_read_cache=True,
        write_cache_format="phasenet",
        log_level=logging.INFO,
    )

    if not mydata.st:
        logger.info("No data associated to event %s", event_id)
    else:
        logger.info(mydata.st.__str__(extended=True))


def _get_data(conf, event_id=None, fdsn_profile=None) -> bool:
    # force eventid_id
    if not event_id:
        event_id = urllib.parse.quote(conf.event_id, safe="")
    else:
        event_id = urllib.parse.quote(event_id, safe="")

    # force fdsn ws
    fdsnws_cfg = conf["fdsnws"]
    if not fdsn_profile:
        default_url_mapping = fdsnws_cfg["default_url_mapping"]
    else:
        default_url_mapping = fdsn_profile
    fdsn_debug = fdsnws_cfg["fdsn_debug"]
    url_mapping = fdsnws_cfg["url_mapping"]

    if default_url_mapping not in fdsnws_cfg["url_mapping"]:
        logger.error("unknown fdsn profile '%s'. Exiting !", default_url_mapping)
        sys.exit(255)

    ws_base_url = url_mapping[default_url_mapping]["ws_base_url"]
    ws_event_url = url_mapping[default_url_mapping]["ws_event_url"]
    ws_station_url = url_mapping[default_url_mapping]["ws_station_url"]
    ws_dataselect_url = url_mapping[default_url_mapping]["ws_dataselect_url"]

    if not event_id:
        logger.error("eventid must be set (in yaml file or using option -e eventid) !")
        sys.exit(255)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not numeric_level:
        logger.error("Invalid loglevel '%s' !", args.loglevel.upper())
        logger.error("loglevel should be: debug,warning,info,error.")
        sys.exit(255)

    output_dirname = os.path.join(conf["output"]["backup_dirname"], event_id)

    mydata = EventFetcher(
        urllib.parse.unquote(event_id),
        starttime=conf["starttime"],
        endtime=conf["endtime"],
        time_length=conf["time_length"],
        starttime_offset=conf["starttime_offset"],
        station_max_dist_km=conf["station_max_dist_km"],
        #
        black_listed_waveforms_id=conf["black_listed_waveforms_id"],
        waveforms_id=conf["waveforms_id"],
        #
        sds=conf["sds"],
        base_url=ws_base_url,
        ws_event_url=ws_event_url,
        ws_station_url=ws_station_url,
        ws_dataselect_url=ws_dataselect_url,
        fdsn_debug=fdsn_debug,
        #
        use_only_trace_with_weighted_arrival=conf[
            "use_only_trace_with_weighted_arrival"
        ],
        keep_only_3channels_station=conf["keep_only_3channels_station"],
        enable_RTrotation=conf["enable_RTrotation"],
        enable_denoising=conf["enable_denoising"],
        backup_dirname=output_dirname,
        enable_write_cache=conf["output"]["enable_write_cache"],
        enable_read_cache=conf["output"]["enable_read_cache"],
        write_cache_format=conf["output"]["write_cache_format"],
        log_level=numeric_level,
    )

    if not mydata.st:
        logger.info("No data associated to event %s", event_id)
        return False
    else:
        logger.info(mydata.event)
        logger.info(mydata.st.__str__(extended=True))
        return True


def load_config(conf_file):
    with open(conf_file, "r") as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            logger.error(e)
            conf = None
    return conf


def denoise_config_check():
    # check if MODEL and CONFIG are defined and exist
    if not os.path.isfile(MODEL):
        logger.error("Deep denoise model file '%s' not found !", MODEL)
        return False
    if not os.path.isfile(CONFIG):
        logger.error("Deep denoise config file '%s' not found !", CONFIG)
        return False
    return True


def denoise_stream(stream, model_name=None, preprocess=True):
    assert model_name in (
        "dae",
        "original",
        "urban",
    ), "Model name must be 'dae', 'original' or 'urban'"

    st = stream.copy()

    if preprocess:
        st.detrend(type="demean")
        st.detrend(type="linear")
        st.taper(max_percentage=0.05, type="cosine", side="both")

    if model_name == "dae":
        try:
            model_dae = load_model(MODEL)
        except ValueError:
            model_dae = load_model(MODEL, compile=False)
        st_denoised, st_noise = denoising_stream(
            stream=st, config_filename=CONFIG, loaded_model=model_dae, parallel=True
        )
    else:
        denoise_model = sbm.DeepDenoiser.from_pretrained(model_name)
        st_denoised = denoise_model.annotate(st)

    # copy coordinates and response to denoised traces
    for tr in st_denoised:
        if model_name in ["original", "urban"]:
            # remove the 'DeepDenoiser_' prefix
            tr.stats.channel = tr.stats.channel.split("_")[1]

        # Find corresponding trace
        mytrace = st.select(id=tr.id)
        assert mytrace, f"Something when wrong when finding {tr.id}"

        tr.stats.coordinates = mytrace[0].stats.coordinates
        tr.stats.response = mytrace[0].stats.response

    return st_denoised


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf",
        default=None,
        dest="conf_file",
        help="eventfetcher configuration file.",
        type=str,
    )

    # use eventid
    parser.add_argument(
        "-e",
        "--eventid",
        default=None,
        dest="eventid",
        help="event id",
        type=str,
    )

    # force fdsn profile
    parser.add_argument(
        "-f",
        "--fdsn-profile",
        default=None,
        dest="fdsn_profile",
        help="fdsn profile",
        type=str,
    )

    # force output directory
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        dest="output_dirname",
        help="output directory",
        type=str,
    )

    # set loglevel
    parser.add_argument(
        "-l",
        "--loglevel",
        default="INFO",
        dest="loglevel",
        help="loglevel (debug,warning,info,error)",
        type=str,
    )
    # add denoise argument (default is False)
    parser.add_argument(
        "-d",
        "--denoise",
        default=False,
        dest="denoise",
        help="enable denoising",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.eventid:
        parser.print_help()
        sys.exit(255)

    # check only if model exist
    if not denoise_config_check():
        sys.exit(255)

    conf = load_config(args.conf_file)
    if not conf:
        sys.exit()

    if args.output_dirname:
        # check if output directory is empty
        if os.path.isdir(args.output_dirname):
            if os.listdir(args.output_dirname):
                logger.error(
                    "output directory '%s' is not empty !", args.output_dirname
                )
                sys.exit(255)

        conf["output"]["backup_dirname"] = args.output_dirname

    if args.denoise:
        conf["enable_denoising"] = True
    else:
        conf["enable_denoising"] = False

    retcode = _get_data(conf, args.eventid, args.fdsn_profile)
    if not retcode:
        sys.exit(1)
    else:
        sys.exit(0)
